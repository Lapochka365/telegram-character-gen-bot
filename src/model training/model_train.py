import numpy as np
import torch
from torch import nn
import pickle
import os.path
import yaml
from model import NeuralNet, one_hot_encode


def get_batches(array, n_sequences, n_steps_window):

    batch_size = n_sequences * n_steps_window
    n_batches = len(array) // batch_size

    array = array[:n_batches * batch_size]
    array = array.reshape((n_sequences, -1))

    for n in range(0, array.shape[1], n_steps_window):

        x = array[:, n:n + n_steps_window]
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, n + n_steps_window]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
        yield x, y


def train_and_validate(
        neural_net,
        encoded_data,
        epochs=50,
        n_sequences=10,
        n_of_steps_in_window=50,
        lr=0.001,
        clip=5,
        val_fraction=0.2,
        cuda=False,
        print_every=50
):

    optimizer = torch.optim.Adam(neural_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_indexes = int(len(encoded_data) * (1 - val_fraction))
    data, val_data = encoded_data[:val_indexes], encoded_data[val_indexes:]

    if cuda:
        neural_net.cuda()

    n_unique_chars = len(neural_net.unique_characters)

    for e in range(epochs):

        # train procedures
        h = neural_net.init_hidden(n_sequences)
        train_losses = []
        neural_net.train()

        for x, y in get_batches(data, n_sequences, n_of_steps_in_window):

            x = one_hot_encode(x, n_unique_chars)
            inputs, targets = torch.from_numpy(x).float(),\
                torch.from_numpy(y).float()

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            neural_net.zero_grad()

            output, h = neural_net.forward(inputs, h)
            target = targets.view(
                n_sequences * n_of_steps_in_window
            ).type(torch.LongTensor)
            if cuda:
                target = target.cuda()
            loss = criterion(output, target)
            loss.backward()

            train_losses.append(loss.item())
            nn.utils.clip_grad_norm_(neural_net.parameters(), clip)

            optimizer.step()

        # validation procedures
        val_h = neural_net.init_hidden(n_sequences)
        val_losses = []
        neural_net.eval()

        with torch.no_grad():

            for x_val, y_val in get_batches(val_data, n_sequences,
                                            n_of_steps_in_window):

                x_val = one_hot_encode(x_val, n_unique_chars)
                x_val, y_val = torch.from_numpy(x_val).float(), \
                    torch.from_numpy(y_val).float()

                val_h = tuple([each.data for each in val_h])

                inputs, targets = x_val, y_val
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = neural_net.forward(inputs, val_h)
                target = targets.view(
                    n_sequences * n_of_steps_in_window
                ).type(torch.LongTensor)
                if cuda:
                    target = target.cuda()
                val_loss = criterion(output, target)

                val_losses.append(val_loss.item())

        if (e + 1) % print_every == 0:
            print('Epoch: {}/{}  | '.format(e + 1, epochs),
                  'Train loss: {:.4f}  | '.format(np.mean(train_losses)),
                  'Valid loss: {:.4f}'.format(np.mean(val_losses)))


if __name__ == '__main__':

    with open('config.yml', 'r', encoding='utf-8') as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    N_HIDDEN = int(config_file['number_of_hidden_layers'])
    N_LSTM_LAYERS = int(config_file['number_of_lstm_layers'])
    N_LINEAR_FEATURES = int(config_file['number_of_features_in_linear'])
    N_SEQS = int(config_file['number_of_sequences'])
    N_STEPS = int(config_file['number_of_steps_per_window'])
    N_EPOCHS = int(config_file['number_of_epochs'])
    LR = float(config_file['learning_rate'])
    VALID_FRACTION = float(config_file['valid_fraction'])
    CUDA = bool(config_file['cuda'])
    PRINT_EVERY_EPOCH = int(config_file['print_every_n_epoch'])

    with open(r'processed data\unique_chars.pkl', 'rb') as f:
        unique_chars = pickle.load(f)

    with open(r'processed data\encoded_text.pkl', 'rb') as f_2:
        encoded_text = pickle.load(f_2)

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

    else:

        net = NeuralNet(
            unique_chars,
            n_hidden=N_HIDDEN,
            n_lstm_layers=N_LSTM_LAYERS,
            n_features_linear=N_LINEAR_FEATURES
        )

    if N_EPOCHS > 0:
        print('>>> The training has been started')
        train_and_validate(
            net,
            encoded_text,
            epochs=N_EPOCHS,
            n_sequences=N_SEQS,
            n_of_steps_in_window=N_STEPS,
            lr=LR,
            val_fraction=VALID_FRACTION,
            cuda=CUDA,
            print_every=PRINT_EVERY_EPOCH
        )
        print('>>> The training is done!')

        checkpoint = {
            'n_hidden': N_HIDDEN,
            'n_lstm_layers': N_LSTM_LAYERS,
            'n_linear_features': N_LINEAR_FEATURES,
            'state_of_model': net.state_dict(),
            'tokens': net.unique_characters
        }

        with open(r'model_states\neural_net_checkpoint.net', 'wb') as f:
            torch.save(checkpoint, f)
        print('>>> Model was successfully saved')
