import json, math, datetime, openai, os, pyarrow, pandas, asyncio
from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
import sparrow_py as kt
from sklearn import preprocessing
import numpy

async def main():
    start = datetime.datetime.now()
    
    # Load user label map
    le = preprocessing.LabelEncoder()
    with open('labels.json') as f:
        le.classes_ = numpy.array(json.load(f))
    
    # Initialize clients
    kt.init_session()
    openai.api_key = os.environ.get("OPEN_AI_KEY")
    slack = SocketModeClient(
        app_token=os.environ.get("SLACK_APP_TOKEN"),
        web_client=WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
    )

    
    # Backfill state with historical data
    messages = kt.sources.PyList(
        rows = pyarrow.parquet.read_table("./messages.parquet").to_pylist(),
        time_column_name = "ts_parsed", 
        key_column_name = "channel",
    )

    
    # Receive Slack messages in real-time
    def handle_message(client, req):
        # Acknowledge the message back to Slack
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        if req.type == "events_api" and "event" in req.payload:
            # ignore message edit, delete, reaction events
            if "previous_message" in req.payload["event"] or req.payload["event"]["type"] == "reaction_added":
                return
            print(f"got message: {req.payload['event']}")
            req.payload["event"]["ts_parsed"] = datetime.datetime.fromtimestamp(float(req.payload["event"]["ts"]))
            del req.payload["event"]["team"]
            messages.add_rows(req.payload["event"])

    slack.socket_mode_request_listeners.append(handle_message)
    slack.connect()
    

    # Compute conversations from individual messages
    conversations = messages.with_key(kt.record({
            "channel": messages.col("channel"),
            "thread": messages.col("thread_ts"),
        })) \
        .select("user", "ts", "text", "reactions") \
        .collect(max=20)


    # Handle each conversation as it occurs
    async for row in conversations.run(materialize=True).iter_rows_async():
        # Skip old messages
        conversation = row["result"]
        if row["_time"] < start or len(conversation) == 0:
            continue
        msg = conversation.pop(-1)
        
        
        # Ask the model who should be notified
        print(f'Starting completion on conversation with first message text: {msg["text"]}')
        res = openai.Completion.create(
            model="davinci:ft-personal:coversation-users-full-kaskada-2023-08-05-14-25-30",
            prompt="start -> " + "\n\n".join([f' {msg["user"]} --> {msg["text"]} ' for msg in conversation]) + "\n\n###\n\n",
            logprobs=5,
            max_tokens=1,
            stop=" end",
            temperature=0.25,
        )

        # Notify interested users
        print(f"Predicted interest logprobs: {res['choices'][0]['logprobs']['top_logprobs'][0]}")
        for user_id in le.inverse_transform(interested_users(res)):
            notify_user(row, msg, user_id, slack)

def interested_users(res):
    users = []
    logprobs = res["choices"][0]["logprobs"]["top_logprobs"][0]
    
    for user in logprobs:
        if math.exp(logprobs[user]) > 0.30:
            user = user.strip()
            if user == "nil":
                continue
            users.append(user.int())
            
    return numpy.array(users)
    
def notify_user(row, msg, user_id, slack):
    app = slack.web_client.users_conversations(
        types="im",
        user=user_id,
    )
    if len(app["channels"]) == 0:
        print(f'User: {user_id} hasn\'t installed the slackbot yet')
        return

    print(f"channel {row['_key']['channel']} message {msg['ts']}")
    link = slack.web_client.chat_getPermalink(
        channel=row["_key"]["channel"],
        message_ts=msg["ts"],
    )

    slack.web_client.chat_postMessage(
        channel=app["channels"][0]["id"],
        text=f'You may be interested in this converstation: <{link["permalink"]}|{msg["text"]}>'
    )

    print(f'Posted alert message to {user_id}')

if __name__ == "__main__":
   asyncio.run(main())