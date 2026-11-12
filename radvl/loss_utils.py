from torch import nn
from radvl.settings_base import parameter_parser
import math
from typing import Any
from torch import Tensor


args = parameter_parser()
import torch
from torch.nn.modules.module import Module


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)

        return (1 - args.alpha) * loss_sum[one_index].sum() + args.alpha * loss_sum[zero_index].sum()


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


class kl_loss(Module):
    def __init__(self, num_edges, num_nodes):
        super(kl_loss, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges


    def forward(self, z_node_log_std, z_node_mean, z_edge_log_std,
                z_edge_mean):
        kl_node = - (0.1 / self.num_nodes) * torch.mean(torch.sum(
            1 + 2 * z_node_log_std - torch.pow(z_node_mean, 2) - torch.pow(torch.exp(z_node_log_std), 2),
            1))

        kl_edge = - (0.1 / self.num_edges) * torch.mean(
            torch.sum(
                1 + 2 * z_edge_log_std - torch.pow(z_edge_mean, 2) - torch.pow(torch.exp(z_edge_log_std), 2), 1))

        kl = kl_node + kl_edge

        return kl


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

