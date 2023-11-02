# BeepGPT - Intelligent notifications powered by Kaskada

BeepGPT keeps you in the loop without disturbing your focus.
Its personalized, intelligent AI continuously monitors your Slack workspace, alerting you to important conversations and freeing you to concentrate on what’s most important.

BeepGPT reads the full history of your (public) Slack workspace and trains a Generative AI model to predict when you need to engage with a conversation.
This training process gives the AI a deep understanding of your interests, expertise, and relationships.
Using this understanding, BeepGPT watches conversations in real-time and notifies you when an important conversation is happening without you.
With BeepGPT you can focus on getting things done without worrying about missing out.

This repo provides a notebook for you to train your own model using your Slack data, as well as a script to run the alerting bot in production.

### Things to do
* To just experiment with Kaskada, feel free to use the [Example Slack Export](#example-slack-export) included in the repo.
* To also experiment with training a model, you will need a OpenAI API key. See [Getting an OpenAI API key](#getting-an-openai-api-key)
* To train a model on your Slack, you will need a json export of your Slack History. An export can be initiated here: https://<your-slack-workspace>.slack.com/services/export. An Admin-level user of your Slack workspace will need to do the export.
* To run the *"Production"* code and receive alerts from a bot, you will need to create a Slack App. See [Creating a Slack App](#creating-a-slack-app)

### Repo Contents

#### Example Slack Export

* `slack-export/` contains an example Slack workspace export. To learn more about the format of the export, see: https://slack.com/help/articles/220556107-How-to-read-Slack-data-exports

Note that some PII data has been removed from the export, but it doesn't effect the files for our use case.

* `slack-export.parquet` contains the data from the example export, in the proper format to be consumed by Kaskada.

#### Generated Slack Data

* The files inside `slack-generation/` include:
    * `notebook.ipynb` a Jupyter notebook which describes how to use OpenAI to generate historical slack data.
    * `projects.json` and `schedule.jsonl` are used in the above notebook.
    * `generated.jsonl` is the raw generated historical data.
* `slack-generation.parquet` contains all the generated slack data in the proper format to be consumed by Kaskada. It is used as example data in the `v2` training notebook below.
* `slack-generation.users.json` contains an example `users.json` file for user lookup in various notebooks.

#### Fine-Tuning Model Training Files

* `FineTuning_v2.ipynb` is a Jupyter notebook which contains all the details of how we successfully trained a model to power BeepGPT.
* `human.py` is a python script used in the v2 training process. See section 2.1 in the `FineTuning_v2.ipynb` notebook for more info.
* `FineTuning_v1.ipynb` is an earlier version of the training procedure. Models trained with this notebook don't generalize as well as those trained with the "v2" notebook.

##### Training Outputs

* `labels_v2.json`
* `labels_v1.json`

These are lists of userIds from the training notebooks. These files are used in the production code to convert single-token user representations back to their original userId.

#### *Production* code

* `beep-gpt.py` contains the code that watches Slack in real-time and alerts you about important conversations. This code uses `messages.parquet` and `labels_.json` as inputs. Note that this code is not production-ready, but functions well enough to demo the full application path.
    1. To run this code, first make sure you using at least Python 3.8. (3.11 recommended):
    1. Next install the required libraries
        ```
        pip install -r requirements.txt
        ```
    1. Then set the following environment variables:
        * `OPEN_AI_KEY` -> Found here: https://platform.openai.com/account/api-keys
        * `SLACK_APP_TOKEN` -> Found here: https://api.slack.com/apps/<your-app-id>/general, should start with `xapp-`
        * `SLACK_BOT_TOKEN` -> Found here: https://api.slack.com/apps/<your-app-id>/oauth, should start with `xoxb-`
    1. Finally start the script
        ```
        python beep-gpt.py
        ```

* `manifest.yaml` contains a template for creating a new App in Slack

#### A New Approach

The following files are related to a new approach we are taking on the project:
* Let users specify what topics they are interested in following.
* The system can recommend topics based on previous history.

Files:
* `ChatCompletion_v1.ipynb` a Jupyter notebook that uses chat completion to determine if a user should be notified.
  * This notebook was primarily created as a baseline for comparing to results of other methods.
* `ChatCompletion_v1_results_*.jsonl` contains the results from the above notebook from various runs.
  * The results are different on each run
* `Vectors_v0.ipynb` a Jupyter notebook that uses embeddings and vector search to determine if a user should be notified.
  * This notebook creates embeddings for the conversations and tries to match them to topics
* `Vectors_v1.ipynb` a Jupyter notebook that evaluates numerous embeddings models to try to determine which works best for this scenario.
  * This notebook create embeddings for the topics and tries to match them to conversations.
  * Outputs are compared to the results from `ChatCompletion_v1.ipynb` to determine which embedding models work best
* `Vectors_v1_<model_name>_scores.jsonl` output from running retrieval against the topic embeddings for each conversation.
  * One file for each embedding model.


#### Conversation Endings

In most of the examples above, we used "10 minutes with no new messages" as the separator between conversations. But in real
life this is probably a bad way to determine if a conversation is ended. In the following files we experiment with the idea
of using few-shot learning and chat completion "in the loop" to determine the end of a conversation.

* `ConversationEnding_v1.ipynb` In this notebook, we use Kaskada to gather up previous messages in a channel and a UDF to call
  the LLM to make the determination. This method includes the past 5 messages in the channel, independent of whether or not they
  are part of the current conversation.
  * `ConversationEnding_v1_results.jsonl` contains the results from this experiment.
  * `ConversationEnding_v1_input.jsonl` contains a small set of data to test if we can determine the end of a conversation based
    on the string output from a previous step
* `ConversationEnding_v2.ipynb` In this notebook, we use [Ray remote Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
  to enable calling the LLM in a parallel manner. This method only include messages that are part of the current conversation.
  * For this method we just proved out the technique. No results file is available.

### Getting an OpenAI API key

In order to experiment with training a model, you will need an OpenAI API key.

If you don't yet have an OpenAI account, go here to sign-up: https://platform.openai.com/signup?launch

After signing up, you will need to add billing details in order to obtain an API key. After doing so, you can create a key here: https://platform.openai.com/account/api-keys

### Creating a Slack App

If you want to run the *Production* code, you will need to create a Slack App and install it into a Slack workspace that you have access to.

1. Start here: https://api.slack.com/apps, and click `Create New App`. Choose `From an App Manifest`.
1. Choose the workspace to install the App in.
1. Copy the contents of `manifest.yaml` and paste it into the window. (make sure to paste it in the `yaml` section)
1. Click `Next`, then `Create`
1. Then on the `Basic Information` page, click `Install to Workspace` and follow the auth flow.
1. Finally, under `App-Level Tokens`, click `Generate Tokens and Scopes`. Add the `connections:write` scope, name it `SocketToken`, and click `Generate`. Don't worry about saving the token somewhere safe, you can always re-access it.

After creating the Slack App, any user that wants to be notified from the App needs to first add it to their personal Apps list. Additionally the App needs to be manually added to any channel you want it to watch.

To add the App to your list: In Slack, in the sidebar, goto `Apps` -> `Manage` -> `Browse Apps`. Click on `BeepGPT` to add it to your app list.

To add the App to a channel: In Slack, go to the channel, click the `v` next to the channel name, goto `Integrations` -> `Apps` -> `Add Apps`. Add `BeepGPT`
