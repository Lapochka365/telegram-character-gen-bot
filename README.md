# telegram-character-gen-bot
### Telegram bot able to generate text messages based on the vk parsed messages.

Just a simple telegram bot that can send text messages generated from vk messages provided.
The workflow is as following:
1. Save your vk messages history into .html files and provide path to them in config.yml (the more messenges, the better)
2. Run vk_messages_parser.py in order to parse the messages into pandas' DataFrame that will be stored in path provided in config.yml
3. Run data_preparation.py to transform the DataFrame into encoded text string, as well as to delete any unnecessary substrings from messages
4. Run model_train.py for model training and validation; all of the hyperparameters can be tuned and accessed via config.yml
5. (Optionally) run generate_message.py to see what kind of the result produces the model
6. Run telegram_bot_run.py after providing token key in config.yml to start the telegram bot
