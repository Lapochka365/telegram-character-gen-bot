import yaml
import telebot
from model_train import NeuralNet
import os.path
import torch
from generate_message import sample
import numpy as np
import emoji
import pickle


def generate_random_message(
        neural_net,
        length_of_message,
        prime_length_min,
        prime_length_max,
        encoded_text_msgs,
        top_k,
        cuda,
        temperature,
        random_emoji
):

    length_total = len(encoded_text_msgs)
    length_prime = np.random.randint(prime_length_min, prime_length_max)
    random_start = np.random.randint(0, length_total - length_prime)
    random_end = random_start + length_prime
    encoded_part_of_text = encoded_text_msgs[random_start:random_end+1]
    primal_sequence = ''
    for encoded_char in encoded_part_of_text:
        primal_sequence += neural_net.int2char[encoded_char]

    message = sample(
        neural_net,
        length_of_message,
        prime_seq_in_msg=primal_sequence,
        top_k=top_k,
        cuda=cuda,
        temperature=temperature
    )

    return message + random_emoji


if __name__ == '__main__':

    with open('config.yml', 'r', encoding='utf-8') as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    TOP_K = int(config_file['top_k_probs'])
    CUDA = bool(config_file['cuda'])
    TEMPERATURE = float(config_file['temperature'])
    API_KEY = str(config_file['api_key'])
    LENGTHS_OF_MESSAGE = config_file['messages_lengths']
    LENGTH_OF_PRIME_MIN = int(config_file['prime_length'][0])
    LENGTH_OF_PRIME_MAX = int(config_file['prime_length'][1])
    GREETING_MESSAGE = str(config_file['greeting_message'])

    with open(r'processed data\encoded_text.pkl', 'rb') as f_2:
        encoded_text = pickle.load(f_2)

    bot = telebot.TeleBot(token=API_KEY)

    if os.path.isfile(r'model_states\neural_net_checkpoint.net'):

        with open(r'model_states\neural_net_checkpoint.net', 'rb') as f:
            model_state = torch.load(f)

        net = NeuralNet(
            unique_tokens=model_state['tokens'],
            n_hidden=model_state['n_hidden'],
            n_lstm_layers=model_state['n_lstm_layers'],
            n_features_linear=model_state['n_linear_features']
        )
        net.load_state_dict(model_state['state_of_model'])

        generated_message = sample(
            net,
            LENGTHS_OF_MESSAGE[0],
            prime_seq_in_msg=GREETING_MESSAGE,
            top_k=TOP_K,
            cuda=CUDA,
            temperature=TEMPERATURE
        )

        @bot.message_handler(commands=['start'])
        def send_greeting_message(message):
            markup = telebot.types.ReplyKeyboardMarkup(
                resize_keyboard=True
            )
            heart_emojis = emoji.emojize(
                ':purple_heart: :revolving_hearts: :sparkling_heart: '
                ':heartbeat: :heartpulse: :two_hearts:'
            ).split()

            rand_heart_emoji = \
                heart_emojis[np.random.randint(0, len(heart_emojis))]
            item_keyboard = telebot.types.KeyboardButton(
                text=f'Напиши мне {rand_heart_emoji}'
            )
            markup.add(item_keyboard)
            bot.send_message(
                message.chat.id,
                text=generated_message,
                reply_markup=markup
            )

        @bot.message_handler(content_types=['text'])
        def generate_new_message(message):
            markup = telebot.types.ReplyKeyboardMarkup(
                resize_keyboard=True
            )
            heart_emojis = emoji.emojize(
                ':purple_heart: :revolving_hearts: :sparkling_heart: '
                ':beating_heart: :growing_heart: :two_hearts:'
            ).split()

            rand_heart_emoji = \
                heart_emojis[np.random.randint(0, len(heart_emojis))]
            item_keyboard = telebot.types.KeyboardButton(
                text=f'Напиши мне {rand_heart_emoji}'
            )
            markup.add(item_keyboard)
            bot.send_message(
                message.chat.id,
                text=generate_random_message(
                    net,
                    LENGTHS_OF_MESSAGE[
                        np.random.randint(0, len(LENGTHS_OF_MESSAGE))
                    ],
                    LENGTH_OF_PRIME_MIN,
                    LENGTH_OF_PRIME_MAX,
                    encoded_text,
                    TOP_K,
                    CUDA,
                    TEMPERATURE,
                    heart_emojis[np.random.randint(0, len(heart_emojis))]
                ),
                reply_markup=markup
            )

        bot.polling()
    else:
        print('>>> You did not train the model, please, run model_train.py')
