import torch.nn as nn
import torch
from utils import SiLU

eps = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class S_net(nn.Module):

    def __init__(self, input_size, hidden_size=50, act_fn='sigmoid', is_explain=False, with_BN=True, with_DO=True, p_dropout=0.25):
        super(S_net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.is_explain = is_explain
        self.with_BN = with_BN
        self.with_DO = with_DO

        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, 1)

        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l2.weight)
        torch.nn.init.kaiming_normal_(self.l3.weight)

        self.l4 = nn.Linear(1, 1, bias=False)

        if with_DO:
            self.dropout = nn.Dropout(p=p_dropout)

        if with_BN:
            self.l1_bn = nn.BatchNorm1d(hidden_size)
            self.l2_bn = nn.BatchNorm1d(hidden_size)

        torch.nn.init.uniform_(self.l4.weight, a=1.4, b=1.40000001)

    def forward(self, x):

        if self.act_fn == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif self.act_fn == 'relu':
            activation = torch.nn.ReLu()
        elif self.act_fn == 'leaky_relu':
            activation = torch.nn.LeakyReLU()
        elif self.act_fn == 'elu':
            activation = torch.nn.ELU()
        elif self.act_fn == 'silu':
            activation = SiLU()

        x = x.view(-1, self.input_size)

        if self.with_BN:
            x = activation(self.l1_bn(self.l1(x)))
            if self.with_DO:
                x = self.dropout(x)
            x = activation(self.l2_bn(self.l2(x)))
            if self.with_DO:
                x = self.dropout(x)
            S = torch.exp(self.l3(x))
        else:
            x = activation(self.l1(x))
            if self.with_DO:
                x = self.dropout(x)
            x = activation(self.l2(x))
            if self.with_DO:
                x = self.dropout(x)
            S = torch.exp(self.l3(x))

        phi = torch.square(self.l4(torch.ones(1, device=device)))

        if self.is_explain:
            return S

        return [S, phi]