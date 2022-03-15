# credit
# https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
from tlstm_node import TSLTM


class TLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # The Stacked T-LSTM node takes input and maps it to hidden states
        cell_list = []
        for layer in range(num_layers):
            if layer == 0:
                cell_list.append(TSLTM(input_dim, hidden_dim[0]))
            else:
                cell_list.append(TSLTM(hidden_dim[layer - 1], hidden_dim[layer]))

        self.cell_list = nn.ModuleList(cell_list)

        # The linear layer which maps from the hidden state to the predicted y
        self.W_y = torch.nn.Linear(hidden_dim[self.num_layers - 1], output_dim)

    def forward(self, inputs, time_deltas):

        bs, seq_sz, _ = inputs.size()
        hidden_sequences_list = []
        for layer in range(self.num_layers):
            hidden_sequences_list.append(torch.empty([bs, seq_sz, self.hidden_dim[layer]])) # (N_layers, n_batches, sequence_length, N_features)

        for layer in range(self.num_layers):
            if layer == 0:
                hidden_sequences_list[layer], (h_T, c_T) = self.cell_list[layer](inputs, time_deltas)
            else:

                hidden_sequences_list[layer], (h_T, c_T) = self.cell_list[layer](hidden_sequences_list[layer - 1].clone(), time_deltas)

        yhat = self.W_y(hidden_sequences_list[layer])  # inputs last hidden_sequence to dense output layers

        return yhat, hidden_sequences_list, (h_T, c_T)
