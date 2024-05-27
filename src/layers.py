import torch
from torch import nn
import torch.nn.functional as F

LAYER_IDS = {}


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)


class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        self.fc_v = nn.Linear(dim, dim)
        self.fc_e = nn.Linear(dim, dim)

    def forward(self, inputs):
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0, 2, 1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)

        # v_output = self.fc_v(v_intermediate)
        # e_output = self.fc_e(e_intermediate)

        return v_output, e_output


class AttentionUnit(nn.Module):
    def __init__(self, dim, channel):
        super(AttentionUnit, self).__init__()
        self.dim = dim
        self.channel = channel
        self.fc_f = nn.Linear(1, channel, bias=False)
        self.fc_g = nn.Linear(1, channel, bias=False)
        self.fc_h = nn.Linear(1, channel, bias=False)
        self.fc_l = nn.Linear(1, channel, bias=False)
        self.fc_m = nn.Linear(channel, 1, bias=False)
        self.fc_n = nn.Linear(channel, 1, bias=False)

        # self.bias_f = Bias(channel)
        # self.bias_g = Bias(channel)
        # self.bias_h = Bias(channel)
        # self.bias_l = Bias(channel)
        self.bias_m = Bias(dim)
        self.bias_n = Bias(dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, inputs):
        # [batch_size, dim]
        v_input, e_input = inputs

        # [batch_size * dim, 1], [batch_size * dim, 1]
        v = torch.reshape(v_input, [-1, 1])
        e = torch.reshape(e_input, [-1, 1])

        # [batch_size, dim, channel], [batch_size, dim, channel]
        f_v = torch.reshape(self.fc_f(v), [-1, self.dim, self.channel])
        g_e = torch.reshape(self.fc_g(e), [-1, self.dim, self.channel])

        # [batch_size, dim, c], [batch_size, dim, c]
        h_v = torch.reshape(self.fc_h(v), [-1, self.dim, self.channel])
        l_e = torch.reshape(self.fc_l(e), [-1, self.dim, self.channel])

        # [batch_size, dim, dim], [batch_size, dim, dim]
        s_matrix = torch.matmul(g_e, f_v.permute(0, 2, 1))
        s_matrix_t = s_matrix.permute(0, 2, 1)

        # [batch_size, dim, dim], [batch_size, dim, dim]
        ev = F.softmax(s_matrix, dim=1)
        ve = F.softmax(s_matrix_t, dim=1)

        # [batch_size * dim, c]
        o_v = torch.reshape(torch.matmul(ev, h_v), [-1, self.channel])
        o_e = torch.reshape(torch.matmul(ve, l_e), [-1, self.channel])

        # [batch_size, dim]
        o_v = torch.reshape(self.fc_m(o_v), [-1, self.dim])
        o_e = torch.reshape(self.fc_n(o_e), [-1, self.dim])

        # output
        v_output = v_input + o_v
        e_output = e_input + o_e

        return v_output, e_output
