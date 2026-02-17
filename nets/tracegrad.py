import warnings
import os
import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct, ElementwiseTensorProduct, TensorProduct

from torch.autograd import grad


class ResidualFullyConnectedLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        """
        Args:
            dim (int): 输入和输出的维度。
            hidden_dim (int): 中间层的隐藏维度，默认为 dim。
        """
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        # 如果输入维度和输出维度不同，使用一个线性投影以实现残差连接
        self.project = nn.Linear(dim, hidden_dim) if dim != hidden_dim else nn.Identity()

    def forward(self, x):
        residual = self.project(x)
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)
        return residual + out

def irreps_mod_concat_length(irreps: Irreps) -> int:
    return sum(mul for mul, ir in irreps)

def irreps_mod_concat(x, irreps: Irreps):
    mods = []
    index = 0
    for mul, ir in irreps:
        dim = ir.dim
        for m in range(mul):
            part = x[:, index:index+dim]
            if dim == 1:
                mods.append(part)
            else:
                mods.append(part.norm(dim=-1, keepdim=True))
            index += dim
    return torch.cat(mods, dim=-1)


class Equi_Nonlin_Grad_Module(nn.Module):
    def __init__(self, irreps_in, z_fea_len=256, hidden_fea_len=1024):
        super(Equi_Nonlin_Grad_Module, self).__init__()
        self.irreps_in = irreps_in
        self.fctp = FullyConnectedTensorProduct(irreps_in, irreps_in, f'{hidden_fea_len}x0e')
        self.nonlinear_layers = nn.Sequential(
            torch.nn.Linear(hidden_fea_len, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(z_fea_len, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(z_fea_len, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            torch.nn.SiLU(),
            nn.Linear(z_fea_len, z_fea_len),
        )

    def forward(self, tensor_in,retain_graph=True):
        with torch.enable_grad():
            x = self.fctp(tensor_in, tensor_in)
            x = self.nonlinear_layers(x)
            y = grad(outputs=x, inputs=tensor_in, grad_outputs=torch.ones_like(x), retain_graph=retain_graph, create_graph=retain_graph, only_inputs=True, 
            allow_unused=False)[0]
        return x, y