#!/usr/bin/env python

import json, math, datetime, openai, os, pyarrow, asyncio, re, time
from datetime import timezone
from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
import kaskada as kd
import pyarrow as pa
import pandas as pd
from sklearn import preprocessing
import numpy

async def main():
    start = datetime.datetime.now(timezone.utc).utcnow()

    # Load user label map
    labels = []
    with open('labels.json') as f:
        labels = json.load(f)

    # Initialize clients
    kd.init_session()
    openai.api_key = os.environ.get("OPEN_AI_KEY")
    slack = SocketModeClient(
        app_token=os.environ.get("SLACK_APP_TOKEN"),
        web_client=WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
    )

    # As messages are received we want to handle them with Kaskada.
    # This creates a data source we can add messages to dynamically.
    # We need to specify a schema, since it can't be inferred from the initial dataset.
    messages = kd.sources.PyDict(
        rows = [],
        schema = pa.schema([
            pa.field("ts", pa.float64()),
            pa.field("channel", pa.string()),
            pa.field("user", pa.string()),
            pa.field("text", pa.string()),
            pa.field("thread_ts", pa.string()),
        ]),
        time_column = "ts",
        key_column = "channel",
        time_unit = "s"
    )

    # Slack's "socket mode" API delivers events in real-time.
    # To use it, we define a handler and register it with Slack's API.
    # As long as this process is running, it will receive live updates.
    def handle_message(client, req):
        # Acknowledge the message back to Slack
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        if req.type == "events_api" and "event" in req.payload:
            # ignore message edit, delete, reaction events
            if "previous_message" in req.payload["event"] or req.payload["event"]["type"] == "reaction_added":
                return
            event = req.payload['event']

            # build message obj with desired properties
            msg = {"thread_ts": None}
            for key in event.keys():
                if key in ["user", "text", "channel"]:
                    msg[key] = event[key]
                elif key in ["ts", "thread_ts"]:
                    msg[key] = float(event[key])

            messages.add_rows(msg) 
    slack.socket_mode_request_listeners.append(handle_message)
    slack.connect()


    async def handle_conversations():
        # For our model's predictions to be accurate, we need to prepare real-time prompts 
        # the same way we prepared training examples.
    
        # Kaskada allows us to use the same operations for historical and real-time data processing.
        conversations = (
            messages.with_key(kd.record({
                "channel": messages.col("channel"),
                "thread": messages.col("thread_ts"),
            }))
            .select("user", "ts", "text")
            .collect(max=20)
        )
                
        # Use the same prompt formatting UDF
        @kd.udf("f<N: any>(x: N) -> string")
        def format_prompts(batch: pd.Series):
            #  Concatenate messages and add a separator
            def format_prompt(conversation):
                msgs = [msg["text"].strip() for msg in reversed(conversation)]
                return "\n\n".join(msgs) + " \n\n###\n\n"
                
            # Apply to each row in the batch
            return batch.map(format_prompt)
            
        # We'll include the structured conversation in the output for building notifications.
        outputs = kd.record({
            "prompt": conversations.pipe(format_prompts),
            "conversation": conversations,
        })
        
        # Now we're ready to continually process conversations as they happen.
        # Kaskada supports "live" execution, where results are incrementally generated as new data arrives.
        # Here, we'll consume each new result as an async generator.
        async for row in outputs.run_iter(kind="row", mode="live"):
            # Send the prompt to the model we fine-tuned
            # We're abusing OpenAI's API a bit here by asking for a single token but 5 "logprobs"
            # (this is why we compressed user-id's to single tokens).
            # The response tells us the 5 most likely next tokens, along with the likelihood of each.
            # We'll use this liklihood as a notification threshold.
            print(f'Starting completion on conversation with prompt: {row["prompt"]}')
            res = openai.Completion.create(
                model="curie:ft-datastax:coversation-next-message-cur-8-2023-08-15-18-01-18",
                prompt=row["prompt"],
                logprobs=5,
                max_tokens=1,
                stop=" end",
                temperature=0,
            )
    
            # Notify interested users (users whose reaction likilihood is above a threshold)
            for user_label in interested_users(res):
                notify_user(
                    channel = row["_key"]["channel"], 
                    msg = row["conversation"].pop(-1), 
                    user_id = labels[user_label], 
                    slack = slack,
                )
                
    await handle_conversations()

def interested_users(res):
    users = []
    logprobs = res["choices"][0]["logprobs"]["top_logprobs"][0]

    for user in logprobs:
        print(f"Predicted user interest for '{user.strip()}': {100 * math.exp(logprobs[user]):.2f}%")
        
        if math.exp(logprobs[user]) > 0.50:
            user = user.strip()
            if user == "nil":
                continue
            users.append(int(user))

    return numpy.array(users)

def notify_user(channel, msg, user_id, slack):
    print(f'Starting notify_user with channel: {channel}, msg: {msg}, user_id: {user_id}')
    try:
        app = slack.web_client.users_conversations(
            types="im",
            user=user_id,
        )
        if len(app["channels"]) == 0:
            print(f'User: {user_id} hasn\'t installed the slackbot yet')
            return

        link = slack.web_client.chat_getPermalink(
            channel=channel,
            message_ts=msg["ts"],
        )

        slack.web_client.chat_postMessage(
            channel=app["channels"][0]["id"],
            text=f'You may be interested in this converstation: <{link["permalink"]}|{msg["text"]}>'
        )

        print(f'Posted alert message to {user_id}')
    except Exception as e:
        print(f'Encountered exception in notify_user {user_id}: {e}')

if __name__ == "__main__":
   asyncio.run(main())
