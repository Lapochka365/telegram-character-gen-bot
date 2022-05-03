# telegram-character-gen-bot
### Telegram bot able to generate text messages based on the vk parsed messages.

Just a simple telegram bot that can send text messages generated from vk messages provided.
The workflow is as following:
1. Save your vk messages history into .html files and provide path to them in _config.yml_ (the more messenges, the better)
2. Run __vk_messages_parser.py__ in order to parse the messages into pandas' DataFrame that will be stored in path provided in _config.yml_
3. Run __data_preparation.py__ to transform the DataFrame into encoded text string, as well as to delete any unnecessary substrings from messages
4. Run __model_train.py__ for model training and validation; all of the hyperparameters can be tuned and accessed via _config.yml_
5. (_Optionally_) run __generate_message.py__ to see what kind of the result produces the model
6. Run __telegram_bot_run.py__ after providing token key in _config.yml_ to start the telegram bot
