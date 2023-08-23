import json, math, datetime, openai, os, pyarrow, asyncio, re
from datetime import timezone
from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
import kaskada as kd
from sklearn import preprocessing
import numpy

def strip_links_and_users(line):
    return re.sub(r"<.*?>", '', line)

def strip_emoji(line):
    return re.sub(r":.*?:", '', line)

def clean_messages(messages):
    cleaned = []
    for msg in messages:
        text = strip_links_and_users(msg)
        text = strip_emoji(text)
        text = text.strip()
        if text == "" or text.find("```") >= 0:
            continue
        cleaned.append(text)
    return cleaned

prompt_suffix = "\n\n###\n\n"
max_prompt_len = 5000

# Format prompt for the OpenAI API
def format_prompt(messages):
    cleaned = clean_messages(messages)
    if len(cleaned) == 0:
        return None
    cleaned.reverse()
    prompt = "\n\n".join(cleaned)
    if len(prompt) > max_prompt_len:
        prompt = prompt[0:max_prompt_len]
    return prompt+prompt_suffix

async def main():
    start = datetime.datetime.now(timezone.utc).utcnow()


    # Load user label map
    le = preprocessing.LabelEncoder()
    with open('labels_.json') as f:
        le.classes_ = numpy.array(json.load(f))

    # Initialize clients
    kd.init_session()
    openai.api_key = os.environ.get("OPEN_AI_KEY")
    slack = SocketModeClient(
        app_token=os.environ.get("SLACK_APP_TOKEN"),
        web_client=WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
    )


    # Backfill state with historical data
    messages = kd.sources.PyList(
        rows = pyarrow.parquet.read_table("messages.parquet").to_pylist(),
        time_column_name = "ts",
        key_column_name = "channel",
        time_unit = "s"
    )


    # Receive Slack messages in real-time
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
            print(f"added message: {msg}")

    slack.socket_mode_request_listeners.append(handle_message)
    slack.connect()


    # Compute conversations from individual messages
    conversations = messages.with_key(kd.record({
            "channel": messages.col("channel"),
            "thread": messages.col("thread_ts"),
        })) \
        .select("user", "ts", "text") \
        .collect(max=2)


    # Handle each conversation as it occurs
    async for row in conversations.run(materialize=True).iter_rows_async():
        # Skip old messages
        conversation = row["result"]
        if row["_time"] < start or len(conversation) == 0:
            continue

        msgs = [msg["text"] for msg in conversation]
        prompt = format_prompt(msgs)
        print(f'Starting completion on conversation with prompt: {prompt}')

        res = openai.Completion.create(
            model="curie:ft-datastax:coversation-next-message-cur-8-2023-08-15-18-01-18",
            prompt=format_prompt(msgs),
            logprobs=5,
            max_tokens=1,
            stop=" end",
            temperature=0,
        )

        msg = conversation.pop(-1)
        channel = row["_key"]["channel"]

        # Notify interested users
        print(f"Predicted interest logprobs: {res['choices'][0]['logprobs']['top_logprobs'][0]}")
        for user_id in le.inverse_transform(interested_users(res)):
            notify_user(channel, msg, user_id, slack)

def interested_users(res):
    users = []
    logprobs = res["choices"][0]["logprobs"]["top_logprobs"][0]

    for user in logprobs:
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
        print(f'Encountered exception in notify_user: {e}')

if __name__ == "__main__":
   asyncio.run(main())
