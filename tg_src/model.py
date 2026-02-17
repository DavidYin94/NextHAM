import warnings
import os
import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.nn import Gate
from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from .from_nequip.cutoffs import PolynomialCutoff
from .from_nequip.radial_basis import BesselBasis
from .from_nequip.tp_utils import tp_path_exists
from .from_schnetpack.acsf import GaussianBasis
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from .e3modules import SphericalBasis, sort_irreps, e3LayerNorm, e3ElementWise, get_random_R, SkipConnection, SeparateWeightTensorProduct, SelfTp
from torch.autograd import grad

epsilon = 1e-8

class Tp_nonlin(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(Tp_nonlin, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = get_gate_stage2(irreps_in1, irreps_in2, irreps_out, act, act_gates)
        irreps_tp_out = self.nonlin.irreps_in
        
        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        self.norm = e3LayerNorm(self.nonlin.irreps_out)
        
        self.irreps_out = self.nonlin.irreps_out

    def forward(self, fea_in1, fea_in2, batch):
        z = self.tp(fea_in1, fea_in2)
        z = self.nonlin(z)
        z = self.norm(z, batch)

        return z

class Equi_Nonlin_Grad_Module(nn.Module):
    def __init__(self, irreps_in, z_fea_len, hidden_fea_len=1024):
        super(Equi_Nonlin_Grad_Module, self).__init__()
        self.fctp = FullyConnectedTensorProduct(irreps_in, irreps_in, f'{hidden_fea_len}x0e')
        self.nonlinear_layers = nn.Sequential(
            nn.Linear(hidden_fea_len, hidden_fea_len),
            torch.nn.LayerNorm(hidden_fea_len, eps=1e-6),
            nn.SiLU(),
            nn.Linear(hidden_fea_len, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            nn.SiLU(),
        )

    def forward(self, tensor_in,retain_graph=True):
        x = self.fctp(tensor_in, tensor_in)
        x = self.nonlinear_layers(x)
        y = grad(outputs=x, inputs=tensor_in, grad_outputs=torch.ones_like(x), retain_graph=retain_graph, create_graph=retain_graph, only_inputs=True, 
        allow_unused=True)[0]
        return x, y


def get_gate_stage2(irreps_in1, irreps_in2, irreps_out, 
                    act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates={1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
            
    irreps_scalars = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    irreps_gated = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
            warnings.warn('Using odd representations as gates')
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir = None
    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_stage2 = Gate(
        irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )
    
    return gate_stage2
        

class EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(EquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_stage2(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        
        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # fully connected net to create tensor product weights
        linear_act = nn.SiLU()
        self.fc = nn.Sequential(nn.Linear(fc_len_in, 64),
                                linear_act,
                                nn.Linear(64, 64),
                                linear_act,
                                nn.Linear(64, self.cfconv.len_weight)
                                )

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        # TODO self-connection here
        return z


class NodeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_out_node, irreps_in_edge,
                 act, act_gates, use_selftp=False, use_sc=True, concat=True, only_ij=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort = sort_irreps(irreps_in1)
                irreps_in1 = self.sort.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node, biases=True)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_stage2(irreps_in1, irreps_in2, irreps_out_node, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_stage2 = False
        else:
            irreps_conv_out = irreps_out_node
            conv_stage2 = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_stage2, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out
        
        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.conv.irreps_out)
            
        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_node, self.irreps_out)
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps

    def forward(self, node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch, selfloop_edge, edge_length):
            
        node_fea_old = node_fea

        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_one_hot)

        node_fea = self.lin_pre(node_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                fea_in = self.sort(fea_in)
            edge_update = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        else:
            edge_update = self.conv(node_fea[index_j], edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        # sigma = 3
        # n = 2
        # edge_update = edge_update * torch.exp(- edge_length ** n / sigma ** n / 2).view(-1, 1)
        
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce='add')

        if self.only_ij:
            node_fea = node_fea + scatter(edge_update[~selfloop_edge], index_j[~selfloop_edge], dim=0, dim_size=node_fea.shape[0], reduce='add')
            
        node_fea = self.lin_post(node_fea)

        if self.use_sc:
            node_fea = node_fea + node_self_connection
            
        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)
            
        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)
            
        node_fea = self.skip_connect(node_fea_old, node_fea)
        
        if self.self_tp is not None:
            node_fea = self.self_tp(node_fea)

        # print(node_fea.shape, edge_update.shape)
        
        return node_fea, edge_update


class EdgeUpdateBlock(nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_stage2(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_stage2 = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_stage2 = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_stage2, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out) # ! consider init_edge
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        
        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])
        
        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)
        
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea


class Invariance_Fea_Extractor(torch.nn.Module):
    def __init__(self, atom_embedded_len, invariance_fea_len, dim_node, dim_edge):
        super(Invariance_Fea_Extractor, self).__init__()
        self.angle_net = torch.nn.Sequential(torch.nn.Linear(1, invariance_fea_len), torch.nn.LayerNorm(invariance_fea_len, eps=1e-6), torch.nn.SiLU(inplace = False), torch.nn.Linear(invariance_fea_len, invariance_fea_len))
        self.invariance_net = torch.nn.Sequential(torch.nn.Linear(3*invariance_fea_len+3*atom_embedded_len, 4*invariance_fea_len+2*invariance_fea_len), torch.nn.LayerNorm(4*invariance_fea_len+2*invariance_fea_len, eps=1e-6), torch.nn.SiLU(inplace = False), torch.nn.Linear(4*invariance_fea_len+2*invariance_fea_len, dim_node))
        self.merge_net = torch.nn.Sequential(torch.nn.Linear(2*dim_node+invariance_fea_len, 2*invariance_fea_len), torch.nn.LayerNorm(2*invariance_fea_len, eps=1e-6), torch.nn.SiLU(inplace = False), torch.nn.Linear(2*invariance_fea_len, dim_edge))

    def forward(self, atom_embedded, edge_length_embedded, edge_vec, edge_index):
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_src_shift = torch.zeros_like(edge_src)
        edge_src_shift[0:-1] = edge_src[1:]
        edge_dst_shift = torch.zeros_like(edge_dst)
        edge_dst_shift[0:-1] = edge_dst[1:]
        # print(edge_src[:10], edge_src_shift[:10], edge_dst[:10], edge_dst_shift[:10])
        # exit(0)
        same_i_diff_j_idx = torch.nonzero((edge_src == edge_src_shift) & (edge_src != edge_dst) & (edge_src_shift != edge_dst_shift), as_tuple=False)[:-1]
        edge_vec_norm = torch.zeros_like(edge_vec)
        edge_vec_norm[same_i_diff_j_idx] = edge_vec[same_i_diff_j_idx]/torch.norm(edge_vec[same_i_diff_j_idx], p=2, dim=-1, keepdim=True)
        edge_vec_norm_shift = torch.zeros_like(edge_vec_norm)
        edge_vec_norm_shift[0:-1] = edge_vec_norm[1:]
        cos_angle_fea = torch.sum(edge_vec_norm.mul(edge_vec_norm_shift), dim=-1, keepdim=True)
        neu_anlge_fea = self.angle_net(cos_angle_fea)
        edge_length_embedded_shift = torch.zeros_like(edge_length_embedded)
        edge_length_embedded_shift[0:-1] = edge_length_embedded[1:]
        invariance_fea_expand = torch.cat((atom_embedded[edge_src], atom_embedded[edge_dst], edge_length_embedded, atom_embedded[edge_dst_shift], edge_length_embedded_shift, neu_anlge_fea), dim = -1)
        invariance_fea_expand_mask = torch.zeros_like(invariance_fea_expand)
        invariance_fea_expand_mask[same_i_diff_j_idx] = invariance_fea_expand[same_i_diff_j_idx]
        invariance_fea_node = self.invariance_net(invariance_fea_expand_mask)
        invariance_fea_node = scatter(invariance_fea_node, edge_dst, dim=0, dim_size=len(atom_embedded))
        invariance_fea_edge = self.merge_net(torch.cat((invariance_fea_node[edge_src], invariance_fea_node[edge_dst], edge_length_embedded), dim = -1))
        return invariance_fea_node, invariance_fea_edge



class Net(nn.Module):
    def __init__(self, num_species, irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                 irreps_post_node, irreps_out_node,irreps_mid_edge, irreps_post_edge, irreps_out_edge, 
                 num_block, r_max, use_sc=True, no_parity=False, use_sbf=True, selftp=False, edge_upd=True,
                 only_ij=False, num_basis=128,
                 act={1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates={1: torch.sigmoid, -1: torch.tanh},
                 if_sort_irreps=False, num_head=4, z_fea_len=1024, trace_out_len=49):
        if no_parity:
            for irreps in (irreps_embed_node, irreps_edge_init, irreps_sh, irreps_mid_node, 
                    irreps_post_node, irreps_out_node,irreps_mid_edge, irreps_post_edge, irreps_out_edge,):
                for _, ir in Irreps(irreps):
                    assert ir.p == 1, 'Ignoring parity but requiring representations with odd parity in net'
        
        super(Net, self).__init__()
        irreps_embed_node = Irreps(irreps_embed_node)
        irreps_post_node = Irreps(irreps_post_node)
        irreps_post_edge = Irreps(irreps_post_edge)
        irreps_mid_node = Irreps(irreps_mid_node)
        irreps_mid_edge = Irreps(irreps_mid_edge)
        irreps_edge_init = Irreps(irreps_edge_init)
        self.num_species = num_species
        self.only_ij = only_ij
        self.num_block = num_block
        self.num_head = num_head
        irreps_embed_node = Irreps(irreps_embed_node)
        assert irreps_embed_node == Irreps(f'{irreps_embed_node.dim}x0e')
        self.embedding = Linear(irreps_in=f"{num_species}x0e", irreps_out=irreps_embed_node)
        self.embedding_node_for_correctrem = Linear(irreps_in=f"{num_species}x0e", irreps_out=irreps_embed_node)
        self.embedding_edge_for_correctrem = Linear(irreps_in=f"{num_species**2}x0e", irreps_out=irreps_edge_init)
        # edge embedding for tensor product weight
        # self.basis = BesselBasis(r_max, num_basis=num_basis, trainable=False)
        # self.cutoff = PolynomialCutoff(r_max, p=6)
        self.basis = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_basis, trainable=False)
        
        # distance expansion to initialize edge feature
        irreps_edge_init = Irreps(irreps_edge_init)
        assert irreps_edge_init == Irreps(f'{irreps_edge_init.dim}x0e')
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=6.0, n_gaussians=irreps_edge_init.dim, trainable=False
        )

        if use_sbf:
            self.sh = SphericalBasis(irreps_sh, r_max)
        else:
            self.sh = SphericalHarmonics(
                irreps_out=irreps_sh,
                normalize=True,
                normalization='component',
            )
        self.use_sbf = use_sbf
        if no_parity:
            irreps_sh = Irreps([(mul, (ir.l, 1)) for mul, ir in Irreps(irreps_sh)])
        else:
            irreps_sh = Irreps(irreps_sh)

        self.irreps_sh = irreps_sh
        
        # self.edge_update_block_init = EdgeUpdateBlock(num_basis, irreps_sh, self.embedding.irreps_out, None, irreps_mid_edge, act, act_gates, False, init_edge=True)
        irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = irreps_edge_init

        self.node_update_blocks = nn.ModuleList([])
        self.edge_update_blocks = nn.ModuleList([])
        self.edge_fea_nonlin_blocks = nn.ModuleList([])

        irreps_atom_embedded = irreps_embed_node
        irreps_edge_embedded = irreps_edge_init
        norm_fea_length = trace_out_len
        for index_block in range(num_block):
            if index_block == num_block - 1:
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_post_node, irreps_edge_prev, act, act_gates, use_selftp=selftp, use_sc=use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_post_edge, act, act_gates, use_selftp=selftp, use_sc=use_sc, if_sort_irreps=if_sort_irreps)
                self.edge_fea_nonlin_blocks.append(Equi_Nonlin_Grad_Module(irreps_post_edge, norm_fea_length))       
            else:
                node_update_block = NodeUpdateBlock(num_species, num_basis, irreps_sh, irreps_node_prev, irreps_mid_node, irreps_edge_prev, act, act_gates, use_selftp=False, use_sc=use_sc, only_ij=only_ij, if_sort_irreps=if_sort_irreps)
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_update_block.irreps_out, irreps_edge_prev, irreps_mid_edge, act, act_gates, use_selftp=False, use_sc=use_sc, if_sort_irreps=if_sort_irreps)
                self.edge_fea_nonlin_blocks.append(Equi_Nonlin_Grad_Module(irreps_mid_edge, norm_fea_length))
            irreps_node_prev = node_update_block.irreps_out
            if edge_update_block is not None:
                irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)
        
        irreps_out_edge = Irreps(irreps_out_edge)
        for _, ir in irreps_out_edge:
            assert ir in irreps_edge_prev, f'required ir {ir} in irreps_out_edge cannot be produced by convolution in the last edge update block ({edge_update_block.irreps_in_edge} -> {edge_update_block.irreps_out})'

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge
        
        self.equi_quantity_decoder = Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge, biases=True)
        self.invar_quantity_decoder = nn.Sequential(
            torch.nn.Linear(norm_fea_length*num_block, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(z_fea_len, z_fea_len),
            torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(z_fea_len, trace_out_len),
        )
        self.nonlin_Z = Tp_nonlin(irreps_node_prev, irreps_node_prev, irreps_node_prev)
        self.lin_Z = Linear(irreps_in=irreps_node_prev, irreps_out=Irreps('1x0e+1x1o+1x2e'), biases=True)
        self.nonlin_E1 = FullyConnectedTensorProduct(irreps_node_prev, irreps_node_prev, Irreps('64x0e'))
        self.nonlin_E2 = nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )
 

    def forward(self, data, is_train=True):

        node_one_hot = F.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        edge_one_hot = F.one_hot(self.num_species * data.x[data.edge_index[0]] + data.x[data.edge_index[1]],
                                 num_classes=self.num_species**2).type(torch.get_default_dtype()) # ! might not be good if dataset has many elements
        
        atom_embedded = self.embedding(node_one_hot)
        node_fea = atom_embedded

        edge_length = data['edge_attr'][:, 0]
        edge_vec = data["edge_attr"][:, [2, 3, 1]] # (y, z, x) order
        invar_feature_list = []

        if self.use_sbf:
            edge_sh = self.sh(edge_length, edge_vec)
        else:
            edge_sh = self.sh(edge_vec).type(torch.get_default_dtype())
        # edge_length_embedded = (self.basis(data["edge_attr"][:, 0] + epsilon) * self.cutoff(data["edge_attr"][:, 0])[:, None]).type(torch.get_default_dtype())
        edge_length_embedded = self.basis(edge_length)
        
        selfloop_edge = None
        if self.only_ij:
            selfloop_edge = torch.abs(data["edge_attr"][:, 0]) < 1e-7

        # edge_fea = self.edge_update_block_init(node_fea, edge_sh, None, edge_length_embedded, data["edge_index"])
        edge_fea = self.distance_expansion(edge_length).type(torch.get_default_dtype())

        node_fea_equi, edge_fea_equi = node_fea, edge_fea
        for run_index_block in range(self.num_block):
            node_update_block, edge_update_block = self.node_update_blocks[run_index_block], self.edge_update_blocks[run_index_block]
            node_fea_equi, _ = node_update_block(node_fea_equi, node_one_hot, edge_sh, edge_fea_equi, edge_length_embedded, data["edge_index"], data.batch, selfloop_edge, edge_length)
            edge_fea_equi = edge_update_block(node_fea_equi, edge_one_hot, edge_sh, edge_fea_equi, edge_length_embedded, data["edge_index"], data.batch)
            invar_nonlin_fea, equi_nonlin_fea = self.edge_fea_nonlin_blocks[run_index_block](edge_fea_equi,retain_graph=is_train)      
            invar_feature_list.append(invar_nonlin_fea)
            edge_fea_equi = edge_fea_equi+equi_nonlin_fea      

        edge_fea_equi = self.equi_quantity_decoder(edge_fea_equi)
        edge_fea_invar = torch.cat(invar_feature_list, dim=-1)
        edge_fea_invar = self.invar_quantity_decoder(edge_fea_invar)
        energy_fea = self.nonlin_E1(node_fea_equi, node_fea_equi)
        energy_fea = self.nonlin_E2(energy_fea)
        zstar_fea = self.nonlin_Z(node_fea_equi, node_fea_equi, data.batch)
        zstar_fea = self.lin_Z(zstar_fea)
        node_fea = torch.cat((energy_fea, zstar_fea), dim = -1)

        return edge_fea_equi

    def __repr__(self):
        info = '===== SO(3)-equivariant Non-linear Representation Learning Framework: ====='
        if self.use_sbf:
            info += f'\nusing spherical bessel basis: {self.irreps_sh}'
        else:
            info += f'\nusing spherical harmonics: {self.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({nupd.irreps_in_node} -> {nupd.irreps_out})'
            if eupd is not None:
                info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput node: ({self.irreps_out_node})'
        info += f'\noutput edge: ({self.irreps_out_edge})'
        
        return info
    
    def analyze_tp(self, path):
        os.makedirs(path, exist_ok=True)
        for index, (nupd, eupd) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            fig, ax = nupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'node_update_{index}.png'))
            fig.clf()
            fig, ax = eupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'edge_update_{index}.png'))
            fig.clf()
