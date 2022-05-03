from model import NeuralNet
import os.path
import torch
import numpy as np
import yaml


def sample(neural_net, size_of_message_in_chars, prime_seq_in_msg,
           top_k=None, cuda=False, temperature=1.0):

    if cuda:
        neural_net.cuda()
    else:
        neural_net.cpu()

    neural_net.eval()

    chars_in_prime = [character for character in prime_seq_in_msg]

    h = neural_net.init_hidden(1)
    char = ''
    for character in prime_seq_in_msg:
        char, h = neural_net.predict(
            character, h, cuda=cuda, top_k=top_k, temperature=temperature
        )

    chars_in_prime.append(char)

    for ii in range(size_of_message_in_chars):
        char, h = neural_net.predict(chars_in_prime[-1], h, cuda=cuda,
                                     top_k=top_k)
        chars_in_prime.append(char)

    return ''.join(chars_in_prime)


if __name__ == '__main__':

    with open('config.yml', 'r', encoding='utf-8') as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    TOP_K = int(config_file['top_k_probs'])
    CUDA = bool(config_file['cuda'])
    TEMPERATURE = float(config_file['temperature'])
    LENGTHS_OF_MESSAGE = config_file['messages_lengths']

    if os.path.isfile(r'model_state\neural_net_checkpoint.net'):

        with open(r'model_state\neural_net_checkpoint.net', 'rb') as f:
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
            LENGTHS_OF_MESSAGE[np.random.randint(0, len(LENGTHS_OF_MESSAGE))],
            prime_seq_in_msg='Привет ',
            top_k=TOP_K,
            cuda=CUDA,
            temperature=TEMPERATURE
        )

        print(generated_message)
