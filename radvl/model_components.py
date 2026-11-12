import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from radvl.encoder_layers import *
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from radvl.attention_layers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed):
    """
    固定随机种子以便结果可复现。
    说明：此函数把 Python / numpy / torch 的随机种子都设置了，但要注意
    CUDA 的确定性还需要设置 torch.backends.cudnn.deterministic = True
    并关闭 benchmark 才能尽量获得可复现的结果（可能牺牲部分性能）。
    """
    random.seed(seed)  # Python 随机库种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python 的哈希随机性（某些环境相关）
    np.random.seed(seed)  # numpy 随机种子
    torch.manual_seed(seed)  # torch CPU 随机种子
    torch.cuda.manual_seed(seed)  # torch 单 GPU 随机种子（如果有 GPU）
    torch.cuda.manual_seed_all(seed)  # torch 多 GPU 随机种子（如果有多个 GPU）


seed_torch(seed=1234)



class HGNN_conv(nn.Module):
    """
    超图卷积实现。
    x: 节点特征矩阵 [N, in_ft]
    G: 邻接矩阵或归一化的超图矩阵 [N, N]

    注意：实现中直接把权重和 G 做矩阵乘法，没有做归一化、安全检查或稀疏加速。
    """
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        # learnable 权重矩阵, shape = (in_ft, out_ft)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重（均匀分布），注意这里用了 out_ft（size(1)）来计算 stdv
        # 常用 fan_in（in_ft）或 kaiming/xavier 初始化，这里可以根据需要替换。
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        # x @ weight: 先做线性变换，再由 G 传播（消息聚合）
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        # 注意：这里假设 G 的形状能与 x 相乘（[N,N] @ [N, out_ft]) -> [N, out_ft]
        return G @ x

class HGCN(nn.Module):
    """
    基于上面单层卷积包装的简单 HGCN（包含激活函数）
    """
    def __init__(self, in_dim, out_dim):
        super(HGCN, self).__init__()
        self.hgc_layer = HGNN_conv(in_dim, out_dim)

    def forward(self, x, G):
        x = self.hgc_layer(x, G)
        x = F.leaky_relu(x, 0.25)
        return x


# ========= 创新型 多头 HGNN Layer ==========
"""
多头版本的 HGNN 卷积：
- 每个 head 有自己的权重矩阵
- 可选的 head 层级（标量）边注意力（use_edge_attention）
- 可选的特征重标定（类似 SE block）
"""
class MultiHeadInnovative_HGNN_conv(nn.Module):

    def __init__(self, in_ft, out_ft, num_heads=8, bias=True, use_edge_attention=True, use_feature_recalibration=True):
        super(MultiHeadInnovative_HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.num_heads = num_heads
        # self.use_edge_attention 是控制是否对超图卷积里的 邻接矩阵 G 加注意力权重。
        self.use_edge_attention = use_edge_attention
        self.use_feature_recalibration = use_feature_recalibration
        """
        self.use_feature_recalibration：是否启用“特征重标定（Feature Recalibration）”机制。
        这个思想来自图像领域的 SE-Block（Squeeze-and-Excitation block），后来被广泛用在图神经网络、Transformer 等地方。
        主要目的：在卷积/聚合之后，模型得到一组特征向量 x_conv，但不同维度的特征对任务的重要性可能不同。
        Feature Recalibration 就是通过一个小的神经网络学习每个特征维度的“权重”，再对原特征进行加权，让“重要维度”被放大，不重要的被压缩。
        """
        # 多头权重
        self.weights = nn.ParameterList([
            Parameter(torch.Tensor(in_ft, out_ft)) for _ in range(num_heads)
        ])

        # 每个 head 独立邻接矩阵（动态调整）
        """
        head 级别的边注意力参数。
        目前实现是每个 head 一个可学习的标量，用来缩放整个邻接矩阵 G。
        这很简单但受限：它无法区分不同边（edge）或不同节点对的注意力。
        """
        self.head_edge_attn = nn.Parameter(torch.Tensor(num_heads)) if use_edge_attention else None

        # SE Block 风格的特征重标定
        if self.use_feature_recalibration:
            self.fc_recalib = nn.Sequential(
                nn.Linear(out_ft, out_ft // 4),
                nn.ReLU(),
                nn.Linear(out_ft // 4, out_ft),
                nn.Sigmoid()
            )

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)

        # 对每个节点特征做 LayerNorm（按最后一维）
        self.norm = nn.LayerNorm(out_ft)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化每个 head 的权重
        for weight in self.weights:
            stdv = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)
        # bias 使用最后一个计算的 stdv（注意：这里 stdv 变量作用域来自循环内，实际上在循环后仍然可见，
        # 但逻辑上有点隐含并不严谨；更好的做法是为 bias 单独计算 stdv）
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.head_edge_attn is not None:
        # 将每个 head 的边缩放初始化为 1
            self.head_edge_attn.data.fill_(1.0)

    def forward(self, x, G):
        """
        x: [N, in_ft]
        G: [N, N] adjacency matrix of hypergraph
        """
        head_outputs = []

        # 对每个 head 进行计算
        for i, weight in enumerate(self.weights):
            # Head-specific 邻接矩阵（当前实现只是把 G 乘以一个标量）
            if self.head_edge_attn is not None:
                G_head = G * self.head_edge_attn[i]  # 动态调节
            else:
                G_head = G

            # 卷积操作：G_head @ (x @ weight)
            out = G_head.matmul(x.matmul(weight))
            head_outputs.append(out)

        # 多头平均（也可以做拼接然后投影）
        x_conv = sum(head_outputs) / self.num_heads

        if self.bias is not None:
            x_conv += self.bias

        # 特征重标定：对每个样本的输出计算缩放系数并点乘
        if self.use_feature_recalibration:
            recalib_weight = self.fc_recalib(x_conv)
            x_conv = x_conv * recalib_weight

        x_conv = self.norm(x_conv)

        return x_conv

# ========= 创新型 HGNN 网络 ==========
"""
由多个 MultiHeadInnovative_HGNN_conv 叠加而成的深层网络。
特点：
- 支持多层堆叠
- 支持 Jumping Knowledge (JK) 通过将每层输出拼接后投影

注意：当前实现中用两层循环创建 layer（num_layers, hidden_list），如果 hidden_list
含有多个不同的 dim，会导致每层输出维度不一致，从而影响 JK 拼接（拼接时维度要一致）。
"""
class Innovative_HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, num_layers=15, num_heads=8, dropout=0.5, use_edge_attention=True, use_feature_recalibration=True, use_jk=True):
        super(Innovative_HGCN, self).__init__()
        self.dropout = dropout
        self.use_jk = use_jk

        self.layers = nn.ModuleList()
        prev_dim = in_dim

        # 构造层：外层循环 num_layers 次，内层遍历 hidden_list
        # 这会生成 num_layers * len(hidden_list) 个子层
        for _ in range(num_layers):
            for hidden_dim in hidden_list:
                self.layers.append(MultiHeadInnovative_HGNN_conv(
                    prev_dim, hidden_dim, num_heads=num_heads,
                    use_edge_attention=use_edge_attention,
                    use_feature_recalibration=use_feature_recalibration
                ))
                prev_dim = hidden_dim

        # 如果使用 JK，需要把所有层的输出拼接后投影回最后一层维度(综合所有层的输出作为特征)
        if self.use_jk:
            self.jk_proj = nn.Linear(len(self.layers) * prev_dim, prev_dim)

    def forward(self, x, G):
        x_embed = x
        all_outputs = []

        # 遍历所有层的HGCN进行特征提取
        for layer in self.layers:
            x_embed_new = layer(x_embed, G)
            x_embed_new = F.leaky_relu(x_embed_new, 0.25)
            x_embed_new = F.dropout(x_embed_new, self.dropout, training=self.training)

            x_embed = x_embed_new

            if self.use_jk:
                all_outputs.append(x_embed_new)

        if self.use_jk:
            # 将所有层的输出在特征维拼接
            x_out = torch.cat(all_outputs, dim=-1)
            x_out = self.jk_proj(x_out)
        else:
            x_out = x_embed

        return x_out


# 这部分属于FastKAN的代码

"""
用于把基函数展开后的向量线性映射到目标维度。继承自 nn.Linear，禁用 bias，并用截断正态分布初始化权重。
"""
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
    def reset_parameters(self) -> None:
        # 用截断正态分布初始化权重（需要 PyTorch 支持该初始化）
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

"""
基于一组 grid 上的径向基函数（高斯），用于把输入投影到基函数空间。
输入 x 形状可能是 [N, input_dim]，输出为 [N, input_dim, num_grids]
"""
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


"""
Fast Kernel Attention/Approximation 层，通过 RBF basis + SplineLinear 实现一类可微基函数映射。
- layernorm: 对输入做归一化
- rbf: 将输入投影为基函数展开
- spline_linear: 把展开后的基函数线性映射到 output_dim
- 可选的 base_update: 使用一个普通的线性变换和激活加残差
"""
class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class NaiveFourierKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (torch.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)

        self.fouriercoeffs = torch.nn.Parameter( torch.randn(2,outdim,inputdim,gridsize) /
                                                (np.sqrt(inputdim) * grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    def forward(self,x):
        xshp = x.shape
        # print(xshp)
        outshape = xshp[0:-1]+(self.outdim,)
        # print(outshape)
        x = torch.reshape(x,(-1,self.inputdim))
        k = torch.reshape( torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) )
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1))
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        y = torch.reshape( y, outshape)
        return y



class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GINLayer, self).__init__()
        self.eps = nn.Parameter(torch.zeros(1))  # 可学习 ε
        self.mlp = nn.Sequential(
            FastKANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                num_grids=8,
                grid_min=-2,
                grid_max=2,
                use_base_update=True
            ),
            # NaiveFourierKANLayer(inputdim=in_dim, outdim=out_dim, gridsize=8),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(out_dim,out_dim)
            # FastKANLayer(
            #     input_dim=64,
            #     output_dim=64,
            #     num_grids=8,
            #     grid_min=-2,
            #     grid_max=2,
            #     use_base_update=True
            # )
        )

    def forward(self, x, edge_index):
        # 聚合邻居特征
        row, col = edge_index
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, x[col])  # sum 聚合

        # GIN 更新规则
        out = self.mlp((1 + self.eps) * x + aggr)
        return out

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(GINEncoder, self).__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, h, edge_index):
        h = self.fc_in(h)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


# 模型总体架构：HGNN+VAE+FastKAN进行特征提取
# 后续会进行改进

class BilinearAttentionFusion(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BilinearAttentionFusion, self).__init__()
        # 双线性权重矩阵
        self.bilinear = nn.Bilinear(dim_in, dim_in, dim_out, bias=False)
        # 投影回原始维度，方便后续处理
        self.proj = nn.Linear(dim_out, dim_in)

    def forward(self, fea1, fea2):
        """
        fea1: [N, d]
        fea2: [N, d]
        return: 融合后的特征 [N, d]
        """
        # 双线性打分 [N, d_out]
        score = self.bilinear(fea1, fea2)

        # 注意力权重 [N, 1]
        attn = torch.sigmoid(torch.sum(score, dim=-1, keepdim=True))

        # 融合特征
        fused = attn * fea1 + (1 - attn) * fea2

        # 投影回 d
        fused = self.proj(fused)

        return fused


import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

#
# # ==========================================
# # 辅助组件 (保持不变)
# # ==========================================
#
# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#
#     def _norm(self, hidden_states: Tensor) -> Tensor:
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         return hidden_states * torch.rsqrt(variance + self.eps)
#
#     def forward(self, hidden_states: Tensor) -> Tensor:
#         return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
#
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
#
#
# # ==========================================
# # 核心类: 交叉差分引导融合
# # ==========================================
#
# class CrossDifferentialGuidedFusion(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads=4, depth=1,
#                  alpha=1.0, tau_base=0.07, mmd_sigma=1.0, gamma1=1.0, gamma2=1.0, beta=0.1):
#         super(CrossDifferentialGuidedFusion, self).__init__()
#
#         # --- 基础配置 ---
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.num_heads = num_heads
#         self.head_dim = dim_in // num_heads
#         self.scaling = self.head_dim ** -0.5
#
#         # --- 损失参数 ---
#         self.alpha = alpha
#         self.tau_base = tau_base
#         self.mmd_kernel_bandwidth = mmd_sigma
#         self.gamma1 = gamma1
#         self.gamma2 = gamma2
#         self.beta = beta
#
#         # ============================================================
#         # 1. 差分注意力投影层 (双向独立参数)
#         # ============================================================
#
#         # 路径 A: fea1 查询 fea2 (Query=1, Key/Value=2)
#         # Q 需要分两组(Signal/Noise) -> 2 * num_heads
#         self.q_proj_1 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
#         self.k_proj_2 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
#         self.v_proj_2 = nn.Linear(dim_in, self.num_heads * self.head_dim, bias=False)
#
#         # 路径 B: fea2 查询 fea1 (Query=2, Key/Value=1)
#         self.q_proj_2 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
#         self.k_proj_1 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
#         self.v_proj_1 = nn.Linear(dim_in, self.num_heads * self.head_dim, bias=False)
#
#         # ============================================================
#         # 2. 差分参数 Lambda (可共享或独立，这里设为独立以获得最大灵活性)
#         # ============================================================
#         self.lambda_init = lambda_init_fn(depth)
#
#         # Set A (For path 1->2)
#         self.lambda_q1_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_k1_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_q2_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_k2_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#
#         # Set B (For path 2->1)
#         self.lambda_q1_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_k1_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_q2_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#         self.lambda_k2_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
#
#         # 输出处理
#         self.out_proj_1 = nn.Linear(self.num_heads * self.head_dim, dim_in, bias=False)
#         self.out_proj_2 = nn.Linear(self.num_heads * self.head_dim, dim_in, bias=False)
#         self.out_norm_1 = RMSNorm(dim_in)
#         self.out_norm_2 = RMSNorm(dim_in)
#
#         # ============================================================
#         # 3. 双向 DGAA 门控组件
#         # ============================================================
#
#         # 门控 A: [Δ1 || fea1 || Z_attn1] -> g1
#         self.g_proj_1 = nn.Linear(3 * dim_in, dim_in)
#         self.g_bias_1 = nn.Parameter(torch.zeros(dim_in))
#
#         # 门控 B: [Δ2 || fea2 || Z_attn2] -> g2
#         self.g_proj_2 = nn.Linear(3 * dim_in, dim_in)
#         self.g_bias_2 = nn.Parameter(torch.zeros(dim_in))
#
#         # 最终融合投影 (将两个方向融合后的结果合并)
#         # 输入维度是 dim_in (来自路径A的融合) + dim_in (来自路径B的融合) = dim_in (求和)
#         self.final_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
#
#         # 可选：双线性增强 (共享或独立，这里简化为不使用双线性以聚焦差分注意力)
#         # self.bilinear = nn.Bilinear(dim_in, dim_in, dim_in, bias=False)
#
#     def _compute_single_direction_diff_attn(self, q_state, k_state, v_state,
#                                             q_proj, k_proj, v_proj,
#                                             l_q1, l_k1, l_q2, l_k2, out_proj, out_norm):
#         """
#         通用差分注意力计算函数 (单向)
#         """
#         N, d = q_state.shape
#
#         # 1. Projections
#         # Q, K split into 2 * heads (Signal vs Noise)
#         q = q_proj(q_state).view(N, 2 * self.num_heads, self.head_dim)
#         k = k_proj(k_state).view(N, 2 * self.num_heads, self.head_dim)
#         v = v_proj(v_state).view(N, self.num_heads, self.head_dim)
#
#         # 2. Transpose for attention: (2*H, N, D)
#         q = q.permute(1, 0, 2)
#         k = k.permute(1, 0, 2)
#         v = v.permute(1, 0, 2)
#
#         q *= self.scaling
#         attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (2*H, N, N)
#         attn_weights = torch.nan_to_num(attn_weights)
#         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
#
#         # 3. Differential Logic (Signal - Lambda * Noise)
#         lambda_1 = torch.exp(torch.sum(l_q1 * l_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(l_q2 * l_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init
#
#         attn_weights = attn_weights.view(self.num_heads, 2, N, N)
#         # Head 0 is Signal, Head 1 is Noise
#         diff_attn_weights = attn_weights[:, 0] - lambda_full * attn_weights[:, 1]  # (H, N, N)
#
#         # 4. Aggregate Value
#         attn_output = torch.matmul(diff_attn_weights, v)  # (H, N, D)
#
#         # 5. Output Project
#         attn_output = attn_output.permute(1, 0, 2).reshape(N, self.num_heads * self.head_dim)
#         output = out_proj(attn_output)
#         output = out_norm(output)
#
#         return output
#
#     def extract_guidance_signals(self, Delta1, g1, Delta2, g2):
#         """
#         提取双向指导信号，取平均值作为整体指导
#         """
#         # 方向1
#         w1 = torch.sigmoid(torch.norm(Delta1, p=2, dim=-1))
#         g_mean1 = torch.mean(g1, dim=-1)
#         s1 = torch.clamp(1 - torch.abs(g_mean1 - 0.5) * 2, min=0.0)
#
#         # 方向2
#         w2 = torch.sigmoid(torch.norm(Delta2, p=2, dim=-1))
#         g_mean2 = torch.mean(g2, dim=-1)
#         s2 = torch.clamp(1 - torch.abs(g_mean2 - 0.5) * 2, min=0.0)
#
#         # 融合策略：平均值 (代表两个模态的一致性共识)
#         w_final = (w1 + w2) / 2
#         s_final = (s1 + s2) / 2
#
#         return w_final, s_final
#
#     # 复用之前的 InfoNCE 和 MMD 函数，保持不变
#     def guided_info_nce(self, fea1, fea2, w):
#         fea1 = F.normalize(fea1, dim=1)
#         fea2 = F.normalize(fea2, dim=1)
#         tau_i = self.tau_base / (1 + self.alpha * w.unsqueeze(-1) + 1e-6)
#         logits = torch.matmul(fea1, fea2.T) / tau_i
#         labels = torch.arange(logits.shape[0], device=logits.device)
#         return F.cross_entropy(logits, labels)
#
#     def guided_mmd(self, fea1, fea2, s):
#         fea1 = F.normalize(fea1, dim=1)
#         fea2 = F.normalize(fea2, dim=1)
#         sigma = self.mmd_kernel_bandwidth
#
#         def gaussian_kernel(a, b):
#             dist_sq = 2 - 2 * torch.matmul(a, b.T)
#             return torch.exp(-dist_sq / (2 * sigma ** 2))
#
#         xx = gaussian_kernel(fea1, fea1)
#         xy = gaussian_kernel(fea1, fea2)
#         yy = gaussian_kernel(fea2, fea2)
#         s_vec = s.unsqueeze(-1)
#         weight_mat = torch.matmul(s_vec, s_vec.T)
#         weight_sum = torch.sum(s) ** 2 + 1e-6
#         loss_xx = torch.sum(weight_mat * xx) / weight_sum
#         loss_xy = torch.sum(weight_mat * xy) / weight_sum
#         loss_yy = torch.sum(weight_mat * yy) / weight_sum
#         mmd = loss_xx + loss_yy - 2 * loss_xy
#         return mmd
#
#     def forward(self, fea1, fea2):
#         """
#         fea1: Modality A (e.g., Similarity features)
#         fea2: Modality B (e.g., Attribute features)
#         """
#
#         # ==========================================
#         # Step 1: 双向差分注意力 (Bidirectional DiffAttn)
#         # ==========================================
#
#         # 路径 A: 1 关注 2 (用 2 去噪 1)
#         # 结果 Z_attn1 是 "在模态2的上下文中看模态1应该是什么样"
#         Z_attn1 = self._compute_single_direction_diff_attn(
#             q_state=fea1, k_state=fea2, v_state=fea2,
#             q_proj=self.q_proj_1, k_proj=self.k_proj_2, v_proj=self.v_proj_2,
#             l_q1=self.lambda_q1_a, l_k1=self.lambda_k1_a,
#             l_q2=self.lambda_q2_a, l_k2=self.lambda_k2_a,
#             out_proj=self.out_proj_1, out_norm=self.out_norm_1
#         )
#
#         # 路径 B: 2 关注 1 (用 1 去噪 2)
#         Z_attn2 = self._compute_single_direction_diff_attn(
#             q_state=fea2, k_state=fea1, v_state=fea1,
#             q_proj=self.q_proj_2, k_proj=self.k_proj_1, v_proj=self.v_proj_1,
#             l_q1=self.lambda_q1_b, l_k1=self.lambda_k1_b,
#             l_q2=self.lambda_q2_b, l_k2=self.lambda_k2_b,
#             out_proj=self.out_proj_2, out_norm=self.out_norm_2
#         )
#
#         # ==========================================
#         # Step 2: 双向差分门控 (Symmetric Differential Gating)
#         # ==========================================
#
#         # 方向 A 差异
#         Delta1 = fea1 - Z_attn1
#         concat_1 = torch.cat([Delta1, fea1, Z_attn1], dim=-1)
#         g1 = torch.sigmoid(self.g_proj_1(concat_1) + self.g_bias_1)
#
#         # 方向 B 差异
#         Delta2 = fea2 - Z_attn2
#         concat_2 = torch.cat([Delta2, fea2, Z_attn2], dim=-1)
#         g2 = torch.sigmoid(self.g_proj_2(concat_2) + self.g_bias_2)
#
#         # ==========================================
#         # Step 3: 对称融合 (Symmetric Fusion)
#         # ==========================================
#
#         # 原始融合逻辑：g * 原特征 + (1-g) * 注意力特征
#         H1 = g1 * fea1 + (1 - g1) * Z_attn1
#         H2 = g2 * fea2 + (1 - g2) * Z_attn2
#
#         # 最终聚合：将两个精炼后的特征相加 (残差思想) 并投影
#         # 这种加法保留了两个视角的信息
#         Z_fused_raw = H1 + H2
#         Z_fused = self.final_proj(Z_fused_raw)
#
#         # ==========================================
#         # Step 4: 损失计算 (集成双向信号)
#         # ==========================================
#
#         # 提取融合后的指导信号 (w 和 s 现在包含了两方面的信息)
#         w, s = self.extract_guidance_signals(Delta1, g1, Delta2, g2)
#
#
#         loss_mmd = self.guided_mmd(fea1, fea2, s)
#         # loss_nce = self.guided_info_nce(fea1, fea2, w)
#
#         # 这里你可以选择是否加上 info_nce 或者正则项
#         self.current_loss = self.gamma1 * loss_mmd
#
#
#         return Z_fused, self.current_loss

# ==========================================
# 基础组件
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# ==========================================
# 核心约束模块（简化版）
# ==========================================

class CrossDiffGuidedConstraint(nn.Module):
    """
    CrossDiff引导的自适应加权约束

    核心思想：
    - 从CrossDiff差分权重中提取"模态互补性"
    - 用此来动态调整两个模态预测的约束强度
    - 互补性强 → 约束弱；互补性弱 → 约束强
    """

    def __init__(self):
        super().__init__()

    def compute_constraint_strength(self, diff_attn_weights):
        """
        从差分注意力权重中提取约束强度

        输入：diff_attn_weights (H, N, N)
              差分后的注意力权重

        输出：constraint_strength (N,) 范围[0,1]
              - 接近1：差异大，约束强
              - 接近0：差异小，约束弱
        """
        # 计算差分权重的平均绝对值（代表差异强度）
        # 平均化所有头和时间步，保留样本维度
        diff_magnitude = torch.mean(
            torch.abs(diff_attn_weights),
            dim=[0, 2]  # 平均化头和目标序列维度 (H, N, N) -> (N,)
        )

        # 正则化到[0, 1]
        min_val = diff_magnitude.min().detach()
        max_val = diff_magnitude.max().detach()

        if max_val > min_val:
            constraint_strength = (diff_magnitude - min_val) / (max_val - min_val + 1e-6)
        else:
            constraint_strength = torch.ones_like(diff_magnitude)

        self.last_strength = constraint_strength.detach()
        return constraint_strength

    def forward(self, H1, H2, diff_attn_weights_1to2, diff_attn_weights_2to1):
        """
        计算CrossDiff引导的自适应约束损失

        参数：
        - H1, H2: 两个模态的融合中间表示 (N, C)
        - diff_attn_weights_1to2: 路径1→2的差分权重 (H, N, N)
        - diff_attn_weights_2to1: 路径2→1的差分权重 (H, N, N)

        返回：
        - loss: 自适应约束损失
        """

        # Step 1: 从两个方向的差分权重中提取约束强度
        strength_1to2 = self.compute_constraint_strength(diff_attn_weights_1to2)  # (N,)
        strength_2to1 = self.compute_constraint_strength(diff_attn_weights_2to1)  # (N,)

        # 取平均作为最终约束强度
        constraint_strength = (strength_1to2 + strength_2to1) / 2  # (N,)
        self.diagnostic_strength = constraint_strength.detach()

        # Step 2: 计算基础约束（两个模态的差异）
        base_constraint = F.mse_loss(H1, H2, reduction='none')  # (N, C)

        # Step 3: 应用自适应加权
        # constraint_strength大 → 约束强 → 损失大
        # constraint_strength小 → 约束弱 → 损失小
        weighted_constraint = constraint_strength.unsqueeze(-1) * base_constraint  # (N, C)

        loss_adaptive = torch.mean(weighted_constraint)

        return loss_adaptive


# ==========================================
# 主融合模块（精简版）
# ==========================================

class CrossDifferentialGuidedFusion(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, depth=1,
                 alpha=1.0, tau_base=0.07, mmd_sigma=1.0, gamma1=1.0):
        super(CrossDifferentialGuidedFusion, self).__init__()

        # 基础配置
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_in // num_heads
        self.scaling = self.head_dim ** -0.5
        self.gamma1 = gamma1

        # ============================================================
        # 1. 差分注意力投影 (双向)
        # ============================================================

        # 路径 A: fea1 查询 fea2
        self.q_proj_1 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj_2 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
        self.v_proj_2 = nn.Linear(dim_in, self.num_heads * self.head_dim, bias=False)

        # 路径 B: fea2 查询 fea1
        self.q_proj_2 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj_1 = nn.Linear(dim_in, 2 * self.num_heads * self.head_dim, bias=False)
        self.v_proj_1 = nn.Linear(dim_in, self.num_heads * self.head_dim, bias=False)

        # ============================================================
        # 2. Lambda 参数（控制信号与噪声的权衡）
        # ============================================================
        self.lambda_init = lambda_init_fn(depth)

        # Set A (for 1->2)
        self.lambda_q1_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2_a = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        # Set B (for 2->1)
        self.lambda_q1_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2_b = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        # 输出处理
        self.out_proj_1 = nn.Linear(self.num_heads * self.head_dim, dim_in, bias=False)
        self.out_proj_2 = nn.Linear(self.num_heads * self.head_dim, dim_in, bias=False)
        self.out_norm_1 = RMSNorm(dim_in)
        self.out_norm_2 = RMSNorm(dim_in)

        # ============================================================
        # 3. 门控融合
        # ============================================================

        self.g_proj_1 = nn.Linear(3 * dim_in, dim_in)
        self.g_bias_1 = nn.Parameter(torch.zeros(dim_in))

        self.g_proj_2 = nn.Linear(3 * dim_in, dim_in)
        self.g_bias_2 = nn.Parameter(torch.zeros(dim_in))

        # 最终投影
        self.final_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

        # ============================================================
        # 4. CrossDiff引导的约束模块
        # ============================================================
        self.constraint = CrossDiffGuidedConstraint()

    def _compute_diff_attn(self, q_state, k_state, v_state,
                           q_proj, k_proj, v_proj,
                           l_q1, l_k1, l_q2, l_k2, out_proj, out_norm):
        """
        计算差分注意力（返回输出和差分权重）
        """
        N, d = q_state.shape

        # 投影
        q = q_proj(q_state).view(N, 2 * self.num_heads, self.head_dim)
        k = k_proj(k_state).view(N, 2 * self.num_heads, self.head_dim)
        v = v_proj(v_state).view(N, self.num_heads, self.head_dim)

        # 转置
        q = q.permute(1, 0, 2)  # (2H, N, d)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)  # (H, N, d)

        # 注意力
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (2H, N, N)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # 差分操作
        lambda_1 = torch.exp(torch.sum(l_q1 * l_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(l_q2 * l_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 分割为信号头和噪声头
        attn_weights = attn_weights.view(self.num_heads, 2, N, N)
        diff_attn = attn_weights[:, 0] - lambda_full * attn_weights[:, 1]  # (H, N, N)

        # 聚合V
        output = torch.matmul(diff_attn, v)  # (H, N, d)

        # 输出投影
        output = output.permute(1, 0, 2).reshape(N, self.num_heads * self.head_dim)
        output = out_proj(output)
        output = out_norm(output)

        return output, diff_attn

    def forward(self, fea1, fea2):
        """
        前向传播

        输入：
        - fea1, fea2: (N, C) 两个模态特征

        输出：
        - Z_fused: 融合特征
        - loss_dict: 包含约束损失的字典
        """

        # ==========================================
        # Step 1: 双向差分注意力
        # ==========================================

        # 路径 A: 1 查询 2
        Z_attn1, diff_attn_1to2 = self._compute_diff_attn(
            q_state=fea1, k_state=fea2, v_state=fea2,
            q_proj=self.q_proj_1, k_proj=self.k_proj_2, v_proj=self.v_proj_2,
            l_q1=self.lambda_q1_a, l_k1=self.lambda_k1_a,
            l_q2=self.lambda_q2_a, l_k2=self.lambda_k2_a,
            out_proj=self.out_proj_1, out_norm=self.out_norm_1
        )

        # 路径 B: 2 查询 1
        Z_attn2, diff_attn_2to1 = self._compute_diff_attn(
            q_state=fea2, k_state=fea1, v_state=fea1,
            q_proj=self.q_proj_2, k_proj=self.k_proj_1, v_proj=self.v_proj_1,
            l_q1=self.lambda_q1_b, l_k1=self.lambda_k1_b,
            l_q2=self.lambda_q2_b, l_k2=self.lambda_k2_b,
            out_proj=self.out_proj_2, out_norm=self.out_norm_2
        )

        # ==========================================
        # Step 2: 差分门控融合
        # ==========================================

        Delta1 = fea1 - Z_attn1
        concat_1 = torch.cat([Delta1, fea1, Z_attn1], dim=-1)
        g1 = torch.sigmoid(self.g_proj_1(concat_1) + self.g_bias_1)

        Delta2 = fea2 - Z_attn2
        concat_2 = torch.cat([Delta2, fea2, Z_attn2], dim=-1)
        g2 = torch.sigmoid(self.g_proj_2(concat_2) + self.g_bias_2)

        # 融合
        H1 = g1 * fea1 + (1 - g1) * Z_attn1
        H2 = g2 * fea2 + (1 - g2) * Z_attn2

        # 最终融合
        Z_fused_raw = H1 + H2
        Z_fused = self.final_proj(Z_fused_raw)

        # ==========================================
        # Step 3: CrossDiff引导的自适应约束
        # ==========================================

        loss_adaptive = self.constraint(H1, H2, diff_attn_1to2, diff_attn_2to1)

        # ==========================================
        # 返回
        # ==========================================

        loss_dict = {
            'adaptive_constraint': loss_adaptive,
            'total_loss': loss_adaptive
        }

        with torch.no_grad():
            self.diagnostic_arrays = {
                'delta_1': Delta1.detach(), 'delta_2': Delta2.detach(),
                'gate_1': g1.detach(), 'gate_2': g2.detach(),
                'alignment_strength': self.constraint.diagnostic_strength,
            }
            self.diagnostics = {
                'delta_squared_mean': (Delta1.square().sum(1) + Delta2.square().sum(1)).mean().item(),
                'gate_1_mean': g1.mean().item(), 'gate_2_mean': g2.mean().item(),
                'alignment_strength_mean': self.constraint.diagnostic_strength.mean().item(),
                'fused_norm_mean': Z_fused.norm(dim=1).mean().item(),
            }
        return Z_fused, loss_adaptive



# ---------- 辅助函数 ----------
def kl_loss_from_stats(mu: torch.Tensor, log_std: torch.Tensor, reduction: str = "sum"):
    """
    计算 diagonal normal KL( N(mu, sigma^2) || N(0,1) ).
    log_std 是 log(sigma).
    """
    sigma2 = torch.exp(2.0 * log_std)
    kl_element = -0.5 * (1.0 + 2.0 * log_std - mu.pow(2) - sigma2)
    if reduction == "sum":
        return kl_element.sum()
    elif reduction == "mean":
        return kl_element.mean()
    else:
        return kl_element

def dynamic_covariance_regularizer(z_mean: torch.Tensor, recon: torch.Tensor, tau: float = 0.1, eps: float = 1e-8):
    """
    动态协同正则化：
    - 计算协方差矩阵 cov (d_z x d_r)
    - 用 cov/std 得到相关系数矩阵 corr (d_z x d_r)
    - alpha = softmax(|corr|/tau, dim=1)  # 每个 z 维度对 r 维度的权重（可改为 dim=0 或全局 softmax）
    - 返回加权协方差: sum(alpha * cov)

    Args:
        z_mean: (N, d_z)
        recon:  (N, d_r)
        tau: temperature for softmax (smaller => sharper weights)
        eps: numerical stability
    Returns:
        scalar cov_reg (torch.scalar)
    """
    z = z_mean.float()
    r = recon.float()
    N = z.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=z.device)

    # 中心化
    zc = z - z.mean(dim=0, keepdim=True)  # (N, d_z)
    rc = r - r.mean(dim=0, keepdim=True)  # (N, d_r)

    # 协方差矩阵 (d_z, d_r)
    cov = (zc.T @ rc) / (N - 1)  # (d_z, d_r)

    # 计算 std
    std_z = torch.sqrt(torch.clamp(zc.var(dim=0, unbiased=False), min=eps))  # (d_z,)
    std_r = torch.sqrt(torch.clamp(rc.var(dim=0, unbiased=False), min=eps))  # (d_r,)

    # outer product denom -> (d_z, d_r)
    denom = std_z.unsqueeze(1) * std_r.unsqueeze(0)  # (d_z, d_r)
    corr = cov / (denom + eps)  # 相关系数矩阵，可能在 [-1,1]

    # 动态权重 alpha：对每个 z 维度在所有 r 维度上做 softmax（行归一化）
    # 可替换 dim=0 做列归一化，或对全矩阵做全局 softmax
    alpha = F.softmax(torch.abs(corr) / (tau + 1e-12), dim=1)  # (d_z, d_r)

    # 加权协方差（保持符号信息）：这里我们保留 cov 的符号（但用 alpha 权重）
    weighted = alpha * cov  # (d_z, d_r)
    cov_reg = torch.sum(torch.abs(weighted))  # 取绝对值和，或直接 torch.sum(weighted) 视你的偏好
    return cov_reg


class RADVL(nn.Module):
    def __init__(self, mi_num, dis_num, hidd_list, num_proj_hidden, hyperpm, num_in_node=218, num_in_edge=271,
                 num_hidden1=512, num_out=64, num_layers=15):
        super(RADVL, self).__init__()

        # 使用FastKAN改进的特征预处理层：把 drug 和 circRNA 两种输入通过基函数映射到相同的隐藏维度
        self.dr_fastkan = FastKANLayer(
            input_dim=mi_num + dis_num,
            output_dim=hidd_list[0],
            num_grids=8,
            grid_min=-2,
            grid_max=2,
            use_base_update=True
        )

        self.circ_fastkan = FastKANLayer(
            input_dim=dis_num + mi_num,
            output_dim=hidd_list[0],
            num_grids=8,
            grid_min=-2,
            grid_max=2,
            use_base_update=True
        )


        self.dropout_rate = hyperpm.dropout
        self.dr_lieanr = nn.Linear(mi_num + dis_num, hidd_list[0])
        self.circ_lieanr = nn.Linear(mi_num + dis_num, hidd_list[0])

        # 两个并行的 HGNN 分支（一个用于 dr，一个用于 circ），共享架构但不共享参数

        # print(111111111111)
        # print(num_layers)
        self.hgcn_dr = Innovative_HGCN(
            in_dim=hidd_list[0],
            hidden_list=hidd_list,
            dropout=self.dropout_rate, # 0.5
            use_edge_attention=True,
            use_feature_recalibration=True,
            use_jk=True,
            num_layers=num_layers
        )
        # print(1111111111111111111111)
        # print(num_layers)
        self.hgcn_circ = Innovative_HGCN(
            in_dim=hidd_list[0],
            hidden_list=hidd_list,
            dropout=self.dropout_rate, # 0.5
            use_edge_attention=True,
            use_feature_recalibration=True,
            use_jk=True,
            num_layers=num_layers
        )


        # 保留原始VAE组件
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, self.dropout_rate).to(device) # dropout_rate:0.3
        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, self.dropout_rate).to(device)
        self.decoder2 = decoder2(act=lambda x: x).to(device)
        self._enc_mu_node = node_encoder1(num_hidden1, num_out, self.dropout_rate, act=lambda x: x).to(device)
        self._enc_log_sigma_node = node_encoder1(num_hidden1, num_out, self.dropout_rate, act=lambda x: x).to(device)
        self._enc_mu_hedge = hyperedge_encoder1(num_hidden1, num_out, self.dropout_rate, act=lambda x: x).to(device)
        self._enc_log_sigma_hyedge = hyperedge_encoder1(num_hidden1, num_out, self.dropout_rate, act=lambda x: x).to(device)

        self.dr_num = 218
        self.circ_num = 271
        self.gin_hidden_dim = hyperpm.hid_dim
        self.gin_layers = 5

        self.dr_gin_encoder = GINEncoder(self.dr_num, self.gin_hidden_dim, self.gin_layers)
        self.circ_gin_encoder = GINEncoder(self.circ_num, self.gin_hidden_dim, self.gin_layers)

        # 改进的特征转换层（保持维度不变）
        self.linear_x = nn.Sequential(
            nn.Linear(hidd_list[-1], 256),
            # nn.Linear(2* hidd_list[-1], 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, hyperpm.hid_dim),
        )

        self.linear_y = nn.Sequential(
            nn.Linear(hidd_list[-1], 256),
            # nn.Linear(2 * hidd_list[-1], 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, hyperpm.hid_dim),
        )

        self.linear_dr_sim = nn.Sequential(
            nn.Linear(self.gin_hidden_dim, 256),
            # nn.Linear(2 * hidd_list[-1], 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
        )

        self.linear_circ_sim = nn.Sequential(
            nn.Linear(self.gin_hidden_dim, 256),
            # nn.Linear(2 * hidd_list[-1], 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),

        )

        # 双线性注意力机制融合
        # self.dr_fusion = BilinearAttentionFusion(hyperpm.hid_dim, hyperpm.hid_dim)
        # self.circ_fusion = BilinearAttentionFusion(hyperpm.hid_dim, hyperpm.hid_dim)

        # 差分注意力引导特征对齐
        self.dr_fusion = CrossDifferentialGuidedFusion(hyperpm.hid_dim, hyperpm.hid_dim)
        self.circ_fusion = CrossDifferentialGuidedFusion(hyperpm.hid_dim, hyperpm.hid_dim)

        # D-CoVAE 超参数：总体强度 co_lambda，softmax 温度 co_tau
        self.co_lambda =hyperpm.co_lambda
        self.co_tau = hyperpm.co_tau

    def sample_latent(self, z_node_enc: torch.Tensor, z_hyperedge_enc: torch.Tensor):
        """
        reparameterization trick，从 encoder 的隐藏表示生成 mu/log_std 和样本。
        """
        z_node_mean = self._enc_mu_node(z_node_enc)
        z_node_log_std = self._enc_log_sigma_node(z_node_enc)  # log(sigma)
        eps_node = torch.randn_like(z_node_mean, device=z_node_mean.device)
        z_node_sample = z_node_mean + torch.exp(z_node_log_std) * eps_node

        z_edge_mean = self._enc_mu_hedge(z_hyperedge_enc)
        z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge_enc)
        eps_edge = torch.randn_like(z_edge_mean, device=z_edge_mean.device)
        z_edge_sample = z_edge_mean + torch.exp(z_edge_log_std) * eps_edge

        return z_node_mean, z_node_log_std, z_node_sample, z_edge_mean, z_edge_log_std, z_edge_sample

    def forward(self, concat_circ_tensor, concat_dr_tensor, G_circ_Kn, G_circ_Km, G_dr_Kn, G_dr_Km, AT, A, dr_sim_graph, circ_sim_graph):
        # drug:218 circRNA:271
        # # 局部特征编码器（通过GCN进行特征提取，单纯预测效果是要好于超图卷积）
        dr_sim_graph = dr_sim_graph.to('cuda:0')
        circ_sim_graph = circ_sim_graph.to('cuda:0')



        # dr_sim_fea = self.dr_gin_encoder(dr_sim_graph.x, dr_sim_graph.edge_index)
        # circ_sim_fea = self.circ_gin_encoder(circ_sim_graph.x, circ_sim_graph.edge_index)
        # print(dr_sim_fea.shape)
        # print(circ_sim_fea.shape)

        # 使用FastKAN进行特征增强
        dr_embedded = self.dr_fastkan(concat_dr_tensor)
        circ_embedded = self.circ_fastkan(concat_circ_tensor)

        # dr_embedded = self.dr_lieanr(concat_dr_tensor)
        # circ_embedded = self.circ_lieanr(concat_circ_tensor)

        # 通过超图卷积
        # dr_feature = self.hgcn_dr(dr_embedded, G_dr_Kn)
        # circ_feature = self.hgcn_circ(circ_embedded, G_circ_Kn)

        dr_hgcn_fea = self.hgcn_dr(dr_embedded, G_dr_Kn)
        circ_hgcn_fea = self.hgcn_circ(circ_embedded, G_circ_Kn)

        # # 特征转换
        dr_hgcn_fea = self.linear_x(dr_hgcn_fea)
        circ_hgcn_fea = self.linear_y(circ_hgcn_fea)
        # print(11111111111111111111111)
        # print(dr_hgcn_fea.shape)
        # print(circ_hgcn_fea.shape)

        dr_sim_fea = self.dr_gin_encoder(dr_sim_graph.x, dr_sim_graph.edge_index)
        circ_sim_fea = self.circ_gin_encoder(circ_sim_graph.x, circ_sim_graph.edge_index)

        # print(2222222222222222222222)
        # print(dr_sim_fea.shape)
        # print(circ_sim_fea.shape)
        # print(dr_sim_fea.shape)
        # print(dr_hgcn_fea.shape)
        dr_sim_fusion, FA_loss = self.dr_fusion(dr_sim_fea, dr_hgcn_fea)
        circ_sim_fusion, FA_loss1 = self.circ_fusion(circ_sim_fea, circ_hgcn_fea)

        FA_loss = (FA_loss + FA_loss1)/2

        # print(dr_sim_fusion.shape)
        # print(circ_sim_fusion.shape)

        # # x = self.linear_x(dr_concat_feature)
        # # y = self.linear_y(circ_concat_feature)

        # dr_sim = self.linear_dr_sim(dr_sim_fea)
        # circ_sim = self.linear_circ_sim(circ_sim_fea)

        # # # 计算异构结果
        # result_h = x.mm(y.t())  # float32
        result_h = circ_sim_fusion.mm(dr_sim_fusion.t())

        # VAE部分保持不变
        z_node_encoder = self.node_encoders1(concat_circ_tensor)  # 注意使用原始输入
        z_hyperedge_encoder = self.hyperedge_encoders1(concat_dr_tensor)
        # self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        self.z_node_mean, self.z_node_log_std, self.z_node_sample, self.z_edge_mean, self.z_edge_log_std, self.z_edge_sample = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        reconstruction_en = self.decoder2(self.z_node_mean, self.z_edge_mean)


        # drug_feature = self.z_edge_mean
        # circRNA_feature = self.z_node_mean
        drug_feature = dr_sim_fusion
        circRNA_feature = circ_sim_fusion

        # print(1111111111111111)
        # print(self.z_node_mean.shape) #[271, 64]
        # print(self.z_edge_mean.shape) #[218, 64]


        result = self.z_node_mean.mm(self.z_edge_mean.t()).float()  # 确保类型一致

        # 将结果压缩到0-1范围
        result_h = torch.sigmoid(result_h)  # 使用sigmoid标准化到(0,1)
        result = torch.sigmoid(result)  # 使用sigmoid标准化到(0,1)

        # print(1111111111111111111)
        # pair_nodes_features0 = torch.cat((circRNA_feature, drug_feature), 0)

        # print(pair_nodes_features0[dr_sim_graph.edge_index].shape)
        # print(pair_nodes_features0[circ_sim_graph.edge_index].shape)
        # print(torch.cat((pair_nodes_features0[dr_sim_graph.edge_index], pair_nodes_features0[circ_sim_graph.edge_index]), 1).shape)
        # 结果融合（保持原有权重）
        score = 0.9 * result + 0.1 * result_h
        self.diagnostic_scores = {'association': result.detach(), 'fused': result_h.detach(), 'combined': score.detach()}
        self.diagnostic_embeddings = {'drug': drug_feature.detach(), 'circ': circRNA_feature.detach()}
        cov_reg_node = dynamic_covariance_regularizer(self.z_node_mean, reconstruction_en, tau=self.co_tau)
        # score = result_h
        # score = result


        return score, reconstruction_en, drug_feature, circRNA_feature, cov_reg_node, FA_loss
