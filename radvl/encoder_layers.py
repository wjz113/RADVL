import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge+271, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        H = H.to(torch.double)
        z1 = self.act(torch.mm(H, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)

class hyperedge_encoder1(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder1, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        H = H.to(torch.double)

        z1 = self.act(torch.mm(H, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)

class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node+218, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H_T):
        H_T = H_T.to(torch.double)


        z1 = self.act(H_T.mm(self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)



class node_encoder1(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder1, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H_T):
        H_T = H_T.to(torch.double)
        z1 = self.act(H_T.mm(self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z





