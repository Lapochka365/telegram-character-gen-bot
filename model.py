import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(
            self,
            unique_tokens,
            number_of_steps_in_window=50,
            n_hidden=256,
            n_lstm_layers=2,
            n_features_linear=1024,
            dropout_prob=0.2,
            lr=0.001
    ):

        super().__init__()
        self.n_steps_window = number_of_steps_in_window
        self.n_hidden = n_hidden
        self.n_lstm_layers = n_lstm_layers
        self.n_features_linear = n_features_linear
        self.dropout_prob = dropout_prob
        self.lr = lr

        self.unique_characters = unique_tokens
        self.int2char = dict(enumerate(self.unique_characters))
        self.char2int = {char: index for index, char in self.int2char.items()}

        self.lstm_layer = nn.LSTM(
            len(self.unique_characters),
            self.n_hidden,
            self.n_lstm_layers,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        self.linear_layer_1 = nn.Linear(
            self.n_hidden, self.n_features_linear
        )
        self.linear_layer_2 = nn.Linear(
            self.n_features_linear, self.n_features_linear
        )
        self.linear_layer_final = nn.Linear(
            self.n_features_linear, len(self.unique_characters)
        )

        self.init_weights()

    def forward(self, x, hc):

        x, (h, c) = self.lstm_layer(x, hc)
        x = self.dropout_layer(x)
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)
        x = F.relu(self.linear_layer_1(x))
        x = F.relu(self.linear_layer_2(x))
        x = self.linear_layer_final(x)

        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None, temperature=1.0):

        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.unique_characters))
        inputs = torch.from_numpy(x).float()
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        output, h = self.forward(inputs, h)

        output = output / temperature
        p = F.softmax(output, dim=1).data
        if cuda:
            p = p.cpu()
        if top_k is None:
            top_chars = np.arange(len(self.unique_characters))
        else:
            p, top_chars = p.topk(top_k)
            top_chars = top_chars.numpy().squeeze()
        p = p.numpy().squeeze()

        id_of_char = np.random.choice(top_chars, p=p / p.sum())

        return self.int2char[id_of_char], h

    def init_weights(self):

        self.linear_layer_1.bias.data.fill_(0)
        self.linear_layer_1.weight.data.uniform_(-1, 1)
        self.linear_layer_2.bias.data.fill_(0)
        self.linear_layer_2.weight.data.uniform_(-1, 1)
        self.linear_layer_final.bias.data.fill_(0)
        self.linear_layer_final.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_sequences):

        weight = next(self.parameters()).data
        return (
            weight.new(self.n_lstm_layers, n_sequences, self.n_hidden).zero_(),
            weight.new(self.n_lstm_layers, n_sequences, self.n_hidden).zero_()
        )


def one_hot_encode(array, number_of_uniq_chars_in_array):

    one_hot_array = np.zeros(
        (np.multiply(*array.shape), number_of_uniq_chars_in_array)
    )
    one_hot_array[np.arange(one_hot_array.shape[0]), array.flatten()] = 1
    one_hot_array = one_hot_array.reshape(
        (*array.shape, number_of_uniq_chars_in_array)
    )

    return one_hot_array
