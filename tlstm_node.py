# credit
# https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/nlp-lstm-byhand.ipynb
# https://github.com/duskybomb/tlstm

import torch
import torch.nn as nn
import math


class TSLTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TSLTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_d = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():  # same init method as PyTorch LSTM implementation
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, time_deltas, init_states=None):
        """
        Implementation of T-LSTM node

        Args:
            input: array of dim (batch_size, seq_length, x_dim) where same seq_length in each batch
            time_deltas: tensor of time_deltas with shape (batch_size, Delta_t)
            init_states: If None then initialize as zeros, if not None ensure correct dimensions

        """
        batch_size, sequence_length, x_dim = inputs.shape

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_dim, requires_grad=False),
                        torch.zeros(batch_size, self.hidden_dim, requires_grad=False))
        else:
            h_t, c_t = init_states

        # For brevity
        exp_1 = torch.exp(torch.tensor(1))
        HS = self.hidden_dim

        hidden_seq = []
        for t in range(sequence_length):
            c_s = torch.tanh(c_t @ self.W_d)
            c_hat_s = c_s * (1 / torch.log(exp_1 + time_deltas[:, t:t + 1])).expand_as(
                c_s)  # expand as ensures the g(Delta_t) are replicated once for each dimension of c_st
            c_l = c_t - c_s
            c_adj = c_l + c_hat_s
            x_t = inputs[:, t, :]  # for all batches take the t'th period
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, cand_mem, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_adj + i_t * cand_mem
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


""" 
OLD VERSION
- not optimized
- should give same results

class TSLTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TSLTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_d = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_f = torch.nn.Linear(input_dim, hidden_dim, bias=False) # Bias term included for U_f, U_i, U_c, U_d
        self.W_i = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_c = torch.nn.Linear(input_dim, hidden_dim, bias=False)

        self.U_f = torch.nn.Linear(hidden_dim, hidden_dim)
        self.U_i = torch.nn.Linear(hidden_dim, hidden_dim)
        self.U_o = torch.nn.Linear(hidden_dim, hidden_dim)
        self.U_c = torch.nn.Linear(hidden_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():  # same init method as PyTorch LSTM implementation
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, time_deltas):
        """"""
        Implementation of T-LSTM node

        Args:
            input: array of dim (batch_size, seq_length, x_dim) where same seq_length in each batch
            time_deltas: tensor of time_deltas with shape (batch_size, Delta_t)

        """"""

        batch_size, sequence_length, x_dim = inputs.shape

        h = torch.zeros(batch_size, self.hidden_dim, requires_grad=False)  # make third dimensions being sequence_length to allow return_sequence=True behaviour
        c = torch.zeros(batch_size, self.hidden_dim, requires_grad=False)
        exp_1 = torch.exp(torch.tensor(1))
        outputs = []
       
        for t in range(sequence_length):
            c_s = torch.tanh(self.W_d(c))
            #one_period_deltas = torch.tensor([time_deltas[b][t:t+1][0] for b in range(batch_size)]).reshape(batch_size, -1)  # torch.tensor(<list shape (3, 1)>).reshape(3,-1)
            #c_hat_s = c_s * (1/torch.log(exp_1 + one_period_deltas)).expand_as(c_s)
            c_hat_s = c_s * (1/torch.log(exp_1 + time_deltas[:, t:t+1])).expand_as(c_s) # expand as ensures the g(Delta_t) are replicated once for each dimension of c_st
            c_l = c - c_s
            c_adj = c_l + c_hat_s
            f = torch.sigmoid(self.W_f(inputs[:, t]) + self.U_f(h))  # bias included in U_f
            i = torch.sigmoid(self.W_i(inputs[:, t]) + self.U_i(h))  # bias included in U_i
            o = torch.sigmoid(self.W_o(inputs[:, t]) + self.U_o(h))  # bias included in U_o
            cand_mem = torch.tanh(self.W_c(inputs[:, t]) + self.U_c(h))  # bias included in U_c
            c = f * c_adj + i * cand_mem
            h = o * torch.tanh(c)
            outputs.append(h)

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs, (h, c)

"""
