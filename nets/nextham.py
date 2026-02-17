import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer
from .tracegrad import ResidualFullyConnectedLayer, Equi_Nonlin_Grad_Module
from torch.nn.parallel import replicate, parallel_apply

# for bessel radial basis

from .graph_attention_transformer import (get_norm_layer, 
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct, SeparableFCTP,
    Vec2AttnHeads, AttnHeads2Vec,
    GraphAttention, FeedForwardNetwork, 
    TransBlock, 
    NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, ScaledScatter
)


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 120 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 72 #18.03065905448718
_AVG_DEGREE = 15.57930850982666



def random_rotation_axis():
    axis = torch.randn(3)  # 生成一个随机的3维向量
    axis = axis / axis.norm()  # 归一化为单位向量
    return axis

def random_rotation_angle():
    return torch.rand(1) * 2 * math.pi  # 生成一个随机角度（0到2π之间）

def rodrigues_rotation_matrix(axis, angle):
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    axis_cross = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = cos_theta * torch.eye(3) + sin_theta * axis_cross + (1 - cos_theta) * torch.outer(axis, axis)
    return rotation_matrix

def random_rotation_matrix():
    axis = random_rotation_axis()
    angle = random_rotation_angle()
    return rodrigues_rotation_matrix(axis, angle)


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py#L111
class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )



class GraphAttentionTransformerHAM(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_edge_embedding = '32x0e+16x1o+8x2e+4x3o+4x4e',
        irreps_edge_output = '1x0e+1x0e+1x1o+1x1o+1x2e+1x0e+1x0e+1x1o+1x1o+1x2e+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e', 
        num_layers=6,
        start_layer=0,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=8.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        use_attn_head=False, 
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None,
        with_trace=True, trace_out_len=49, use_w2v=True):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.use_attn_head = use_attn_head
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.lmax = 5
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.irreps_edge_mid = o3.Irreps(irreps_edge_embedding)
        self.irreps_edge_output = o3.Irreps(irreps_edge_output)

        self.use_w2v = use_w2v

        self.ele_embed = torch.nn.Embedding(num_embeddings = 120, embedding_dim = 32)

        self.basis_type = basis_type

        self.rbf = GaussianExpansion(0, 10, num_basis=64)
        # print(self.irreps_edge_output)
        # exit(0)
        self.input_rs = LinearRS(self.irreps_edge_output, self.irreps_edge_mid, bias=False)
        
        self.edge_lin = LinearRS(o3.Irreps(f'32x0e+{self.irreps_edge_output.sort()[0].simplify()}'), self.irreps_edge_output)
          
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        print('self.alpha_drop, self.proj_drop, self.out_drop, self.drop_path_rate', self.alpha_drop, self.proj_drop, self.out_drop, self.drop_path_rate)
        if self.use_attn_head:
            self.head = GraphAttention(irreps_node_input=self.irreps_feature, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=o3.Irreps('1x0e'),
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop)
        else:
            self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
            )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.start_layer = start_layer
        self.norm_fea_length = 256
        self.with_trace = with_trace
        self.blocks = torch.nn.ModuleList()
        self.edge_feature_nonlin_block = torch.nn.ModuleList()
        
        self.build_blocks()

        if self.with_trace:
            
            self.trace_out_len = trace_out_len
            
            num_block = len(self.edge_feature_nonlin_block)
            # print(self.norm_fea_length, num_block, len(self.edge_feature_nonlin_block))
            # exit(0)
            gradnorm_fea_len = 512
            self.nonlinear_layers_for_norm_reg = torch.nn.Sequential(
                    torch.nn.Linear(self.norm_fea_length*num_block, gradnorm_fea_len),
                    torch.nn.SiLU(),
                    torch.nn.Linear(gradnorm_fea_len, gradnorm_fea_len),
                    torch.nn.SiLU(),
                    torch.nn.Linear(gradnorm_fea_len, gradnorm_fea_len),
                    torch.nn.SiLU(),
                    torch.nn.Linear(gradnorm_fea_len, self.trace_out_len)                
            )


        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i == (self.num_layers - 1):
                irreps_node_input = self.irreps_edge_mid
                irreps_edge_attr = self.irreps_edge_mid
                irreps_node_output = self.irreps_feature
                irreps_edge_output = o3.Irreps(f'32x0e+{self.irreps_edge_output.sort()[0].simplify()}')
            elif i == 0:
                if self.use_w2v:
                    irreps_node_input =  o3.Irreps('32x0e')
                    irreps_edge_attr =  o3.Irreps('64x0e')
                else:
                    irreps_node_input =  self.irreps_edge_mid
                    irreps_edge_attr =  self.irreps_edge_mid
                irreps_node_output = self.irreps_edge_mid
                irreps_edge_output = self.irreps_edge_mid     
            else:
                irreps_node_input = self.irreps_edge_mid
                irreps_edge_attr = self.irreps_edge_mid
                irreps_node_output = self.irreps_edge_mid
                irreps_edge_output = self.irreps_edge_mid
                

            blk = TransBlock(irreps_node_input=irreps_node_input, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr, 
                irreps_node_output=irreps_node_output,
                irreps_edge_output=irreps_edge_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=irreps_node_input, 
                num_heads=self.num_heads, 
                irreps_pre_attn=irreps_node_input, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=irreps_node_input,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)

            if i != self.num_layers -1:
                hidden_fea_len = 1024
            else:
                hidden_fea_len = self.irreps_feature.dim
            if i >= self.start_layer and self.with_trace:
                if i == (self.num_layers - 1):
                    tracegrad_module = Equi_Nonlin_Grad_Module(o3.Irreps(f'32x0e'), z_fea_len=self.norm_fea_length, hidden_fea_len=hidden_fea_len)                
                else:
                    tracegrad_module = Equi_Nonlin_Grad_Module(irreps_edge_output, z_fea_len=self.norm_fea_length, hidden_fea_len=hidden_fea_len)
                self.edge_feature_nonlin_block.append(tracegrad_module)
            

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    # the gradient of energy is following the implementation here:
    # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L186
    @torch.enable_grad()
    def forward(self, weak_ham_in, node_num, edge_src, edge_dst, edge_vec, batch, node_atom, use_sep = True, range_dis = [0.0, 10.0]):
        # ---------------------- Input Features ---------------------- #
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr, x=edge_vec[:, [1,2,0]], normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        mask_dis = torch.logical_and(edge_length >= range_dis[0], edge_length < range_dis[1]).float().unsqueeze(1)
        edge_features_invar_list = []
        if self.use_w2v:
            node_features = self.ele_embed(node_atom)
            edge_features = torch.cat((node_features[edge_src], node_features[edge_dst]), dim = -1)
        else:
            edge_features = self.input_rs(weak_ham_in) 
            mask = torch.logical_and(edge_src == edge_dst, edge_length < 1e-5)
            filtered_edge_indices = edge_src[mask]
            filtered_edge_features = edge_features[mask]
            node_features = scatter(filtered_edge_features, filtered_edge_indices, dim=0, dim_size=node_num)

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        for blk_idx, blk in enumerate(self.blocks):
            node_features, edge_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_features, edge_sh = edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch, edge_length=edge_length)
            if blk_idx >= self.start_layer:
                if blk_idx < len(self.blocks)-1:
                    edge_features_invar, edge_features_equi_nonlin = self.edge_feature_nonlin_block[blk_idx - self.start_layer](edge_features)
                    scale = (torch.linalg.norm(edge_features, 2, dim=-1, keepdim=True)/(torch.linalg.norm(edge_features_equi_nonlin, 2, dim=-1, keepdim=True)+1e-5)).detach()
                    edge_features = edge_features + 0.2 * scale * edge_features_equi_nonlin 
                else:
                    edge_features_invar, edge_features_equi_nonlin = self.edge_feature_nonlin_block[blk_idx - self.start_layer](edge_features[:,:32]) 
                edge_features_invar_list.append(edge_features_invar)
        # ---------------------- Output Decoding ---------------------- #
        edge_ham = self.edge_lin(edge_features)
        if self.with_trace:
            edge_features_invar =  torch.cat(edge_features_invar_list, dim=-1)
            edge_trace  = self.nonlinear_layers_for_norm_reg(edge_features_invar)
        else:
            edge_trace = 0.
        if use_sep:
            edge_ham = edge_ham * mask_dis * torch.exp(-0.2*edge_length)[:, None]  
            edge_trace = edge_trace * mask_dis
        return edge_ham, edge_trace, mask_dis


@register_model
def graph_attention_transformer_nonlinear_materials_ham_soc(irreps_in, irreps_edge, radius, num_basis=64, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, use_w2v = True, **kwargs):
    model = GraphAttentionTransformerHAM(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_edge_embedding='32x0e+16x1o+16x1e+8x2e+8x2o+8x3e+8x3o+8x4e', num_layers=4, start_layer=start_layer,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e+1x5o',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e+8x4o+1x5o+1x6e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='32x0e+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e+8x4o+1x5o+1x6e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        trace_out_len=trace_out_len, with_trace=with_trace, use_w2v = use_w2v)
    return model

class GaussianExpansion(torch.nn.Module):
    def __init__(self, min_val, max_val, num_basis):
        super(GaussianExpansion, self).__init__()
        self.centers = torch.linspace(min_val, max_val, num_basis).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        return torch.exp(-((x - self.centers) ** 2) / (2 * self.width ** 2))