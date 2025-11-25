"""
Simplified ASTGCN implementation adapted for music composition analysis.

This module implements a simplified version of the Attention-based Spatial-Temporal
Graph Convolutional Network (ASTGCN) for composer classification from polyphonic music.

Original ASTGCN Paper:
    Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019).
    Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.
    AAAI Conference on Artificial Intelligence, 33(01), 922-929.
    https://doi.org/10.1609/aaai.v33i01.3301922

Adaptations for music:
    - Domain: Traffic networks → Musical graph structures (MIDI pitch nodes)
    - Task: Regression (flow prediction) → Classification (composer identification)
    - Graph: Static spatial → Dynamic polyphonic voice connections
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    get_laplacian,
    subgraph,
    mask_to_index,
    index_to_mask,
)

GLOBAL_DROPOUT = 0.5
DROPOUT_2D= 0.4

def get_output_size(seq_len, kernel_size, stride, dilation, transpose=False):
    kernel_size = kernel_size*dilation-1
    padding = kernel_size//2
    if not transpose:
        return  math.ceil(((seq_len - kernel_size + 2 * padding) / stride) + 1)
    else:
        return round((seq_len-1)*stride - 2*padding + (kernel_size-1) + 1)
        



class ChebConvAttention(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator with attention from the
    `Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/316161>`_ paper
    :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "mean")
        super(ChebConvAttention, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"
        self.Kval = K
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._normalization = normalization
        self._weight = Parameter(torch.Tensor(K, in_channels, out_channels)) 

        if bias:
            self._bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("_bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._weight)
        if self._bias is not None:
            nn.init.uniform_(self._bias)

    #--forward pass-----
    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_attr: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        edge_index, edge_attr = get_laplacian(
            edge_index, edge_attr, 'sym', dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_attr = (2.0 * edge_attr) / lambda_max
        edge_attr.masked_fill_(edge_attr == float("inf"), 0)

        edge_index, edge_attr = add_self_loops(   
            edge_index, edge_attr, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_attr is not None

        return edge_index, edge_attr #for example 307 nodes as deg, 340 edges , 307 nodes as self connections

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        spatial_attention: torch.FloatTensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the ChebConv Attention layer (Chebyshev graph convolution operation).

        Arg types:
            * x (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in).
            * edge_index (Tensor array) - Edge indices.
            * spatial_attention (PyTorch Float Tensor) - Spatial attention weights, with shape (B, N_nodes, N_nodes).
            * edge_attr (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * batch (PyTorch Tensor, optional) - Batch labels for each edge.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.

        Return types:
            * out (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, F_out).
        """
        '''if self._normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )'''

        

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_attr,
            self._normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )
        row, col = edge_index # refer to the index of each note each is a list of nodes not a number # (954, 954)
        Att_norm = norm * spatial_attention[:, row, col] # spatial_attention for example (32, 307, 307), -> (954) * (32, 954) -> (32, 954)
        num_nodes = x.size(self.node_dim) #for example 307
        # (307, 307) * (32, 307, 307) -> (32, 307, 307) -permute-> (32, 307,307) * (32, 307, 1) -> (32, 307, 1)
        TAx_0 = torch.matmul(
            (torch.eye(num_nodes, device=x.device, dtype=x.dtype) * spatial_attention).permute(
                0, 2, 1
            ),
            x,
        ) #for example (32, 307, 1)
        out = torch.matmul(TAx_0, self._weight[0]) #for example (32, 307, 1) * [1, 64] -> (32, 307, 64)
        edge_index_transpose = edge_index[[1, 0]]
        if self._weight.size(0) > 1:
            TAx_1 = self.propagate(
                edge_index_transpose, x=TAx_0, norm=Att_norm, size=None
            )
            out = out + torch.matmul(TAx_1, self._weight[1])

        for k in range(2, self._weight.size(0)):
            TAx_2 = self.propagate(edge_index_transpose, x=TAx_1, norm=norm, size=None)
            TAx_2 = 2.0 * TAx_2 - TAx_0
            out = out + torch.matmul(TAx_2, self._weight[k])
            TAx_0, TAx_1 = TAx_1, TAx_2

        if self._bias is not None:
            out += self._bias

        return out #? (b, N, F_out) (32, 307, 64)

    def message(self, x_j, norm):
        if norm.dim() == 1:  # true
            return norm.view(-1, 1) * x_j  # (954, 1) * (32, 954, 1) -> (32, 954, 1)
        else:
            d1, d2 = norm.shape
            return norm.view(d1, d2, 1) * x_j

    def __repr__(self):
        return "{}({}, {}, K={}, normalization={})".format(
            self.__class__.__name__,
            self._in_channels,
            self._out_channels,
            self._weight.size(0),
            self._normalization,
        )





class SpatialAttention(nn.Module):
    r"""An implementation of the Spatial Attention Module (i.e compute spatial attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/316161>`_
    
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, 
                 q_channels: int,
                 q_nodes: int,
                 q_timesteps: int,
                 k_channels: int = None, 
                 k_nodes: int = None,
                 k_timesteps: int = None,
                 n_heads = 1,
                 time_strides: int = 1):
        super(SpatialAttention, self).__init__()
        self.n_heads = 1
        self.q_channels = q_channels
        self.q_nodes = q_nodes
        self.q_timesteps = q_timesteps

        self.k_channels = k_channels if k_channels is not None else q_channels
        self.k_timesteps = k_timesteps if k_timesteps is not None else q_timesteps
        self.k_nodes = k_nodes if k_nodes is not None else q_nodes
        self.num_time_chunks = 1
        
        

        self.k_nodes = k_nodes
        
        #print(f'chunk_sizes: {self.chunk_sizes}')
        #self._W1alt = nn.ParameterList([nn.Parameter(torch.FloatTensor(chunk)) for chunk in self.chunk_sizes]) #for example (12)])
        self._W1 = nn.Parameter(torch.FloatTensor(self.q_timesteps))  # for example (12)
        self._W2 = nn.Parameter(
            torch.FloatTensor(self.q_channels,self.k_timesteps)
        )  # for example (1, 12)
        self._W3 = nn.Parameter(torch.FloatTensor(self.k_channels))  # for example (1)
        self._bs = nn.Parameter(
            torch.FloatTensor(1, self.k_nodes, self.q_nodes)
        )  # for example (1,307, 307)
        self._Vs = nn.Parameter(
            torch.FloatTensor(self.k_nodes,self.q_nodes )
        )  # for example (307, 307)


        self.directional_mask = torch.triu(torch.ones([self.k_nodes, self.q_nodes], requires_grad=False), diagonal=1).to(bool)


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                #nn.init.kaiming_uniform_(p)
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_self: torch.FloatTensor, X_other: torch.FloatTensor, 
                node_mask: torch.FloatTensor=None,
                time_mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **S** (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        # lhs = left hand side embedding;
        # to calculcate it : 
        # multiply with W1 (B, N, F_in, T)(T) -> (B,N,F_in)
        # multiply with W2 (B,N,F_in)(F_in,T)->(B,N,T)
        # for example (32, 307, 1, 12) * (12) -> (32, 307, 1) * (1, 12) -> (32, 307, 12) 
        #print(torch.matmul(X_self, self._W1).shape)
        #X_selflist = torch.split(X_self, self.self_channels//self.n_heads, -2)

        denom = math.sqrt(self.q_channels)

        q_channel_scale = math.sqrt(self.q_channels)
        k_channel_scale = math.sqrt(self.k_channels)
        q_ts_scale = 1
        k_ts_scale = 1



  
        #print(f'X_self shape: {X_self.shape}, X_other shape: {X_other.shape}, W1 shape: {W1.shape}, W2 shape: {W2.shape}, W3 shape: {W3.shape}, denom: {denom}')
        first_matmul = torch.matmul(X_self, self._W1) / q_ts_scale
        #print(f'first_matmul shape: {first_matmul.shape}')
        LHS = torch.matmul(first_matmul, self._W2) / q_channel_scale

        # rhs = right hand side embedding
        # to calculcate it : 
        # mutliple W3 with X (F)(B,N,F,T)->(B, N, T) 
        # transpose  (B, N, T)  -> (B, T, N)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12) -transpose-> (32, 12, 307)
        RHS = torch.matmul(self._W3, X_other).transpose(-1, -2) / k_ts_scale

        '''RHS = RHS.masked_fill(~time_mask.view(-1,self.self_timesteps,1), 0.0) if time_mask is not None else RHS
        RHS = RHS.masked_fill(~node_mask.view(-1, 1,self.num_of_vertices), 0.0) if node_mask is not None else RHS'''

        
        
        #print(f'space LHS mean/std: {LHS.mean()}/{LHS.var()}, max = {LHS.max()}, min = {LHS.min()} {LHS.shape}')
        #print(f'space RHS mean/std: {RHS.mean()}/{RHS.var()}, max = {RHS.max()}, min = {RHS.min()}, {RHS.shape}')
        # Then, we multiply LHS with RHS : 
        # (B,N,T)(B,T, N)->(B,N,N)
        # for example (32, 307, 12) * (32, 12, 307) -> (32, 307, 307) 
        # Then multiply Vs(N,N) with the output
        # (N,N)(B, N, N)->(B,N,N) (32, 307, 307)
        # for example (307, 307) *  (32, 307, 307) ->   (32, 307, 307)

        sigOut = F.sigmoid((torch.matmul(LHS, RHS)/k_channel_scale)) # (B, N, N) for example (32, 307, 307)
        #print(f'space RHS mean/std: {sigOut.mean()}/{sigOut.var()}, max = {sigOut.max()}, min = {sigOut.min()}, {sigOut.shape}')
        '''sigOutOnesCount = torch.sum(sigOut == 1.0)
        sigOutZeroCount = torch.sum(sigOut == 0.0)
        print(f'space sigOut mean/std: {sigOut.mean()}/{sigOut.var()}, max = {sigOut.max()}, min = {sigOut.min()}, sigout ones count: {sigOutOnesCount}, sigout zero count: {sigOutZeroCount}')'''
        S = torch.matmul(self._Vs, sigOut) / q_ts_scale


        S = F.softmax(S, dim=-1)
            
        S = F.dropout(S, p=GLOBAL_DROPOUT, training=self.training)

        #print(f'after softmax space S mean/std: {S.mean()}/{S.var()}, max = {S.max()}, min = {S.min()}')
        return S # (B,N,N) for example (32, 307, 307)


class TemporalAttention(nn.Module):
    r"""An implementation of the Temporal Attention Module( i.e. compute temporal attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/316161>`_
    
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, 
                 q_channels: int,
                 q_nodes: int,
                 q_timesteps: int,
                 k_channels: int = None,
                 k_nodes: int = None,
                 k_timesteps: int = None,
                 causal_mask: Optional[bool] = False,
                 n_heads=1):

        super(TemporalAttention, self).__init__()
        self.n_heads = n_heads
        self.causal_mask = causal_mask
        self.q_channels = q_channels
        self.q_nodes = q_nodes
        self.q_timesteps = q_timesteps
        self.k_channels = k_channels if k_channels is not None else q_channels
        self.k_nodes = k_nodes if k_nodes is not None else q_nodes
        self.k_timesteps = k_timesteps if k_timesteps is not None else q_timesteps

        '''self._U1 = nn.Parameter(torch.FloatTensor(n_heads, num_of_vertices))  # for example 307
        self._U2s = nn.ParameterList([ nn.Parameter(torch.FloatTensor(self_channels//n_heads, num_of_vertices)) for head in range(n_heads)])
        #self._U2 =  nn.Parameter(torch.FloatTensor(1, self_channels, num_of_vertices))
        #self._U2 = nn.Parameter(torch.FloatTensor(self_channels, num_of_vertices))
        self._U3 = nn.Parameter(torch.FloatTensor(n_heads, other_channels//n_heads))  # for example (1)
        #self._U3 = nn.Parameter(torch.FloatTensor(other_channels)) # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(n_heads,other_timesteps, self_timesteps)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(torch.FloatTensor(n_heads, other_timesteps, other_timesteps))  #for example (12, 12)'''
        self.register_buffer(
            'forward_add_mask',
            torch.zeros(self.q_timesteps, self.q_timesteps, requires_grad=False)
            .masked_fill(torch.triu(torch.ones(self.q_timesteps, self.q_timesteps, requires_grad=False), diagonal=1).to(bool), -torch.inf)
        )
        self.register_buffer(
            'forward_add_bool_mask',
            torch.triu(torch.ones(self.q_timesteps, self.q_timesteps, requires_grad=False), diagonal=1).to(bool)
        )
        self.directional_mask = torch.triu(torch.ones([self.q_timesteps, self.q_timesteps], requires_grad=False), diagonal=1).to(bool)

        self._U1 = nn.Parameter(torch.FloatTensor(self.q_nodes))  # for example 307
        self._U2 = nn.Parameter(torch.FloatTensor(self.q_channels, self.k_nodes)) #for example (1, 307)
        self._U3 = nn.Parameter(torch.FloatTensor(self.k_channels))  # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(1, self.q_timesteps, self.q_timesteps)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(torch.FloatTensor(self.k_timesteps, self.q_timesteps))  #for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('sigmoid'))
            else:
                nn.init.uniform_(p)

    def forward(self, X_self: torch.FloatTensor, 
                X_other: torch.FloatTensor, 
                node_mask: torch.BoolTensor=None,
                time_mask: Optional[torch.BoolTensor] = None
                ) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **E** (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
    

        q_channel_scale = math.sqrt(self.q_channels)
        k_channel_scale = math.sqrt(self.k_channels)
        q_node_scale = 1
        k_node_scale = 1
        # lhs = left hand side embedding;
        # to calculcate it : 
        # permute x:(B, N, F_in, T) -> (B, T, F_in, N)  
        # multiply with U1 (B, T, F_in, N)(N) -> (B,T,F_in)
        # multiply with U2 (B,T,F_in)(F_in,N)->(B,T,N)
 

        #print(f'x: {X_self.shape}, U1: {self._U1.shape}, U2: {self._U2.shape}, U3: {self._U3.shape}, denom: {denom}')

        #U1 = self._U1.masked_fill(~(node_mask.to(torch.float).sum(0).to(torch.bool)).view(self.num_of_vertices), 0.0) if node_mask is not None else self._U1
        #U2 = self._U2.masked_fill(~(node_mask.to(torch.float).sum(0).to(torch.bool)).view(1, self.num_of_vertices), 0.0) if node_mask is not None else self._U2
        LHS = torch.matmul(torch.matmul(X_self.permute(0,3, 2, 1),self._U1)/q_node_scale, self._U2) / q_channel_scale
        RHS = torch.matmul(self._U3, X_other) / k_node_scale # (32, 307, 12)
        '''LHS = LHS.masked_fill(~time_mask.view(-1, self.self_timesteps, 1), 0.0) if time_mask is not None else LHS
        LHS = LHS.masked_fill(~node_mask.view(-1, 1, self.num_of_vertices), 0.0) if node_mask is not None else LHS
        
        RHS = RHS.masked_fill(~time_mask.view(-1,1, self.self_timesteps), 0.0) if time_mask is not None else RHS
        RHS = RHS.masked_fill(~node_mask.view(-1, self.num_of_vertices, 1), 0.0) if node_mask is not None else RHS'''

        ##print(f'time LHS mean/std: {LHS.mean()}/{LHS.var()}, max = {LHS.max()}, min = {LHS.min()},{LHS.shape}')
        #print(f'time RHS mean/std: {RHS.mean()}/{RHS.var()}, max = {RHS.max()}, min = {RHS.min()},{RHS.shape}')
        # Them we multiply LHS with RHS : 
        # (B,T,N)(B,N,T)->(B,T,T)
        # for example (32, 12, 307) * (32, 307, 12) -> (32, 12, 12) 
        # Then multiply Ve(T,T) w
        # ith the output
        # (T,T)(B, T, T)->(B,T,T)
        # for example (12, 12) *  (32, 12, 12) ->   (32, 12, 12)




        '''Ve = self._Ve.masked_fill(~time_mask.view(-1, 1, self.self_timesteps), 0.0) if time_mask is not None else self._Ve
        Ve = Ve.masked_fill(~time_mask.view(-1, self.self_timesteps, 1),0.0) if time_mask is not None else self._Ve
        be = self._be.masked_fill(~time_mask.view(-1, 1,self.self_timesteps), 0.0) if time_mask is not None else self._be
        be = be.masked_fill(~time_mask.view(-1, self.self_timesteps, 1), 0.0) if time_mask is not None else self._be'''
        sigOut = F.sigmoid((torch.matmul(LHS, RHS)/k_channel_scale)) #  + self._be
        #sigOutOnesCount = torch.sum(sigOut == 1.0)
        #print(f'time sigOut mean/std: {sigOut.mean()}/{sigOut.var()}, max = {sigOut.max()}, min = {sigOut.min()}, sigout shape: {sigOut.shape}')
     
        E = torch.matmul(self._Ve, sigOut)/q_channel_scale
        if E.shape[1:] == (self.q_timesteps, self.q_timesteps):
            add_mask = self.forward_add_mask.view(-1, self.q_timesteps, self.q_timesteps) if self.causal_mask else torch.zeros_like(E)  #
            #add_mask = add_mask.masked_fill(~time_mask.view(-1, self.q_timesteps, 1), -torch.inf) if time_mask is not None else add_mask
            E += add_mask

        #
            #
        #if add_mask.shape[0] == E.shape[0]:
        
        #print(f'time E mean/std: {E.mean()}/{E.var()}, max = {E.max()}, min = {E.min()}')
        E = F.softmax(E, dim=-2) #  (B, T, T)  for example (32, 12, 12)
        #E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0) # replace nan and inf with 0.0
        #if time_2d_dropout_mask is not None: 
        #E = F.dropout(E, p=1-MASK_DENSITY, training=self.training)
        
        E = E.masked_fill(~time_mask.view(-1, 1, self.q_timesteps),0.0) if time_mask is not None else E
        E = E.masked_fill(~time_mask.view(-1, self.q_timesteps, 1),0.0) if time_mask is not None else E

        E = F.dropout(E, p=GLOBAL_DROPOUT, training=self.training)
        #print(f'time E after softmax mean/std: {E.mean()}/{E.var()}, max = {E.max()}, min = {E.min()}')
        return E # (B, T, T) for example (32, 12, 12)

class STAttention(nn.Module):
    def __init__(
        self,
        self_channels: int,
        out_channels: int,
        num_of_vertices: int,
        self_timesteps: int,
        time_kernel: int = 1,
        is_transpose: bool = False,
        causal_mask: Optional[bool] = False,
        k=1

    ):
        super(STAttention, self).__init__()
        self.self_timesteps = self_timesteps
        self.is_transpose = is_transpose
        self.num_of_vertices = num_of_vertices
        self.self_channels = self_channels
        self.out_channels = out_channels
        self.causal_mask = causal_mask
        self.time_kernel = time_kernel
        self.padding = time_kernel
        self._normalization = "sym"
        self._temporal_attention = TemporalAttention(
            self_channels,
            num_of_vertices,
            self_timesteps, causal_mask=causal_mask
        )

        
        
        self._spatial_attention = SpatialAttention(
            self_channels,
            num_of_vertices,
            time_kernel,
            k_channels=self_channels,
            k_nodes=num_of_vertices,
            k_timesteps=time_kernel,
        )

        self.layer_norm = nn.LayerNorm(self_channels, elementwise_affine=True)

        self._chebconv_attention = ChebConvAttention(
            self_channels, out_channels, k, self._normalization,
                    bias=False
        )

        self.dense_gcn = DenseGCNConv(self_channels, out_channels, bias=False)

    def forward(
        self,
        X: torch.FloatTensor,
        input_graphs,  # Can be List[Data] or a single Batch (global graph)

    ) -> torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **input_graphs**: Either a list of graphs (one per timestep) or a single Batch (global graph reused for all timesteps)

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape # (32, 307, 1, 12)

        # Check if we have a single global graph or per-timestep graphs
        use_global_graph = isinstance(input_graphs, Batch)

        X_tilde = self._temporal_attention(X, X, node_mask=None, time_mask=None)
        X_tilde = torch.matmul(X.reshape(batch_size, -1, self.self_timesteps), X_tilde.transpose(-1,-2))
        X_tilde_out = X_tilde.reshape(batch_size, num_of_vertices, num_of_features, -1)

        if not self.is_transpose:
            padding = (self.time_kernel - 1) // 2
            X_tilde_out = F.pad(X_tilde_out, (padding, padding))
        else:
            padding = (self.time_kernel - 1)
            X_tilde_out = F.pad(X_tilde_out, (padding, 0))

        lambda_max = None
        X_hat = []

        X = X.reshape([1, batch_size*num_of_vertices, num_of_features, -1])

        # Pre-compute graph info if using global graph (compute once, reuse)
        if use_global_graph:
            graph_t = input_graphs
            # For global graph, compute per-sample node masks based on which nodes have edges
            # We need to work with each sample in the batch separately
            per_sample_masks = []
            per_sample_edge_info = []
            for b in range(batch_size):
                # Get edges for this sample in the batch
                sample_mask = graph_t.batch == b if graph_t.batch is not None else None
                if sample_mask is not None:
                    sample_edge_mask = sample_mask[graph_t.edge_index[0]]
                    sample_edges = graph_t.edge_index[:, sample_edge_mask]
                    sample_edge_attr = graph_t.edge_attr[sample_edge_mask] if graph_t.edge_attr is not None else None
                    # Adjust edge indices to be relative to sample (not batch)
                    offset = b * num_of_vertices
                    sample_edges = sample_edges - offset
                else:
                    sample_edges = graph_t.edge_index
                    sample_edge_attr = graph_t.edge_attr

                xmask_b = index_to_mask(torch.unique(sample_edges), size=num_of_vertices)
                k_b = xmask_b.count_nonzero()
                subset_b = mask_to_index(xmask_b)
                edge_index_b, edge_attr_b = subgraph(subset_b, sample_edges, sample_edge_attr, num_nodes=num_of_vertices, relabel_nodes=True)
                per_sample_masks.append((xmask_b, k_b))
                per_sample_edge_info.append((edge_index_b, edge_attr_b))

        for t in range(X.shape[-1]):
            x_chunk = X[:, :, :, t]
            att_chunk = X_tilde_out[:, :, :, t:t+self.time_kernel]
            X_tilde_t = self._spatial_attention(att_chunk, att_chunk)

            if use_global_graph:
                # Process each sample in batch separately with its own mask
                outblocks = []
                for b in range(batch_size):
                    xmask_b, k_b = per_sample_masks[b]
                    edge_index_b, _ = per_sample_edge_info[b]

                    x_chunk_b = x_chunk[:, b*num_of_vertices:(b+1)*num_of_vertices, :]
                    x_chunk_masked = x_chunk_b[:, xmask_b, :]

                    # Get attention for this sample
                    att_b = X_tilde_t[b:b+1]  # (1, N, N)
                    node_mask_b_2d = (xmask_b.unsqueeze(-1).float() @ xmask_b.unsqueeze(0).float()).bool()
                    att_matrix = att_b[0][node_mask_b_2d].reshape(k_b, k_b).unsqueeze(0)

                    if edge_index_b.numel() > 0:
                        outblock_b = self._chebconv_attention(x_chunk_masked, edge_index_b, att_matrix, edge_attr=None, lambda_max=lambda_max, batch=None)
                    else:
                        outblock_b = self.dense_gcn(x_chunk_masked, att_matrix)

                    # Scatter back to full size
                    full_out = torch.zeros(num_of_vertices, self.out_channels, device=outblock_b.device, dtype=outblock_b.dtype)
                    full_out[xmask_b] = outblock_b.squeeze(0)
                    outblocks.append(full_out)

                outblock = torch.stack(outblocks, dim=0)  # (batch_size, num_vertices, out_channels)
            else:
                graph_t_curr = input_graphs[t]
                graph_t_curr.validate()
                xmask = index_to_mask(torch.unique(graph_t_curr.edge_index), size=x_chunk.shape[1])
                k = xmask.count_nonzero()
                node_mask_t_2d = (xmask.unsqueeze(-1).float() @ xmask.unsqueeze(0).float()).bool()
                x_chunk_masked = x_chunk[:, xmask, :]
                att_matrix = torch.block_diag(*X_tilde_t)[node_mask_t_2d].reshape(k, k).unsqueeze(0)
                subset = mask_to_index(xmask)
                edge_index, edge_attr = subgraph(subset, graph_t_curr.edge_index, graph_t_curr.edge_attr, num_nodes=xmask.shape[0], relabel_nodes=True)
                batch = graph_t_curr.batch[xmask] if graph_t_curr.batch is not None else None
                outblock = self._chebconv_attention(x_chunk_masked, edge_index, att_matrix, edge_attr=None, lambda_max=lambda_max, batch=batch) if edge_index.numel() > 0 else self.dense_gcn(x_chunk_masked, att_matrix)
                outblock = torch.masked_scatter(torch.zeros(num_of_vertices*batch_size, self.out_channels, device=outblock.device, dtype=outblock.dtype), xmask.unsqueeze(-1), outblock)
                outblock = outblock.view(batch_size, num_of_vertices, self.out_channels)

            outblock = outblock.unsqueeze(-1)  # (batch_size, num_vertices, out_channels, 1)
            X_hat.append(outblock)

        X_hat = F.gelu(torch.cat(X_hat, dim=-1))
        X_hat = X_hat.view([batch_size, num_of_vertices, X_hat.shape[-2], -1])
        return X_hat




class ASTGCNBlock(nn.Module):
    r"""An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/316161>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        K: int,
        self_nodes: int,
        out_nodes: int,
        self_labels: int,
        self_space_channels_in: int,
        self_space_channels_out: int,
        self_time_kernel: int,
        self_time_strides: int,
        self_timesteps_in: int,
        self_timesteps_out: int,
        self_time_channels_in: int,
        self_time_channels_out: int,
        cross_nodes: int,
        cross_channels: int,
        cross_timesteps: int,
        temporal_conv: Optional[bool] = False,
        causal_mask: Optional[bool] = False
    ):
        super(ASTGCNBlock, self).__init__()
        self.padding = self_time_kernel-1
        self.K = K
        self.self_nodes = self_nodes
        self.out_nodes = out_nodes
        self.self_labels = self_labels
        self.self_space_channels_in = self_space_channels_in
        self.self_space_channels_out = self_space_channels_out
        self.time_kernel = self_time_kernel
        self.self_time_strides = self_time_strides
        self.self_timesteps_in = self_timesteps_in
        self.self_timesteps_out = self_timesteps_out
        self.self_time_channels_in = self_time_channels_in
        self.self_time_channels_out = self_time_channels_out
        self.cross_nodes = cross_nodes
        self.cross_channels = cross_channels
        self.cross_timesteps = cross_timesteps
        self.temporal_conv = temporal_conv
        #print(f'time_att_strides: {time_att_strides}, time_kernel: {time_kernel}, time_dilations: {time_dilations}')
        self.st_att = STAttention(self.self_space_channels_in,
                                  self.self_space_channels_out,
                                  self.self_nodes,
                                  self.self_timesteps_in,
                                  time_kernel=self.time_kernel,
                                  k=self.K,
                                  causal_mask=causal_mask
                                  )
        
        #if self.self_timesteps_in < self.self_timesteps_out:
        self.st_clusters = STAttention(self.self_space_channels_in,
                                    self.out_nodes,
                                    self.self_nodes,
                                    self.self_timesteps_out,
                                    time_kernel=self.time_kernel,
                                    k=self.K,
                                    causal_mask=causal_mask
                                    )
        '''else:
            self.st_clusters = STAttention(self.self_space_channels_in,
                                    self.out_nodes,
                                    self.self_nodes,
                                    self.self_timesteps_in,
                                    time_kernel=self.time_kernel,
                                    k=self.K,
                                    causal_mask=causal_mask
                                    )'''
        self._time_convolution = nn.Conv2d(
                self.self_time_channels_in,
                self.self_time_channels_out,
                kernel_size=(1, self.time_kernel),
                stride=(1, self.self_time_strides),
                padding=(0, 0),
                bias=False,
                groups=self.self_time_channels_in
            )
        self._time_convolution_clusters = nn.Conv2d(
                self.out_nodes,
                self.out_nodes,
                kernel_size=(1, self.time_kernel),
                stride=(1, self.self_time_strides),
                padding=(0, 0),
                bias=False
            )
        self._residual_convolution = nn.Conv2d(
            self.self_space_channels_in, self.self_time_channels_out, kernel_size=(1, 1), stride=(1, self.self_time_strides),
            bias=False, groups=self.self_space_channels_in
        )
        self._layer_norm = nn.LayerNorm(self.self_time_channels_out, bias=False)

    def forward(
        self,
        X: torch.FloatTensor,
        input_graphs: List[Data],
        output_graphs: List[Data] = None,
        pool_first: bool = True
    ) -> torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.x

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        batch_size, num_of_vertices, num_of_features, input_timesteps = X.shape # (32, 307, 1, 12)
        self.output_graphs = []
        
        
        self.pooling_loss = 0.0

        X_hat = self.st_att(X, input_graphs)

        
        if X.shape[1] != X_hat.shape[1]:
            X = F.interpolate(X.permute(0,3,2,1), [num_of_features, self.out_nodes])
            X = X.permute(0,3,2,1)

        if self.self_time_strides == 1:
            if X_hat.shape[-1] != self.self_timesteps_out:
                X_hat = F.interpolate(X_hat, [num_of_features, self.self_timesteps_out])
            if X.shape[-1] != self.self_timesteps_out:
                X = F.interpolate(X, [num_of_features, self.self_timesteps_out])
        #F.pad(X_hat, (((self.time_conv_kern)*self.dilation_factor)-1, 0))
        X_hat = F.pad(X_hat, (self.padding, 0))
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3)) #[:,:,:,-self.timesteps_out:] # will give (32, 64, 307,12)
        #X_hat = F.dropout(X_hat, p=GLOBAL_DROPOUT, training=self.training)
        # (b,N,F,T)-permute>(b,F,N,T) (1,1)->(b,F,N,T)  (32, 64, 307, 12)
        

            
        X = self._residual_convolution(X.permute(0, 2, 1, 3))     # will also give (32, 64, 307,12)
        #-adding X + X_hat->(32, 64, 307, 12)-premuting-> (32, 12, 307, 64)-layer_normalization_-premuting->(32, 307, 64,12) 


        

        X = self._layer_norm(F.gelu(X+X_hat).permute(0, 2, 3, 1))
        
        X = X.permute(0, 1, 3, 2)
        return X, input_graphs, input_graphs, 0.0 # (b,N,F,T) for example (32, 307, 64,12)









    

class PolyphonyGCN(torch.nn.Module):

    def __init__(self, config):

        super(PolyphonyGCN, self).__init__()
        blocknums = list(range(config.num_blocks))
        self.config = config
        self.input_dim = config.input_dim
        self.num_labels = config.num_labels
        self.periods = config.periods
        self.num_layers = config.num_blocks
        self.make_positional_sinusoids()
        self.src_embedding = nn.Embedding(config.num_labels, config.input_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(config.num_labels, config.input_dim, padding_idx=0)
        self.encoder = torch.nn.ModuleList(
            [
                ASTGCNBlock(
                    config.K,
                    config.nodes_in[i],
                    config.nodes_out[i],
                    config.num_labels,
                    config.input_dim,
                    config.input_dim,
                    config.time_kernels[i],
                    config.strides[i],
                    config.periods_in[i],
                    config.periods_out[i],
                    config.input_dim,
                    config.input_dim,
                    config.num_nodes,
                    config.input_dim,
                    config.time_kernels[i],
                    temporal_conv=True,
                    causal_mask=True
                )
                for i in blocknums
            ]
        )
        
        blocknums.reverse()
        self.layer_pos_emb = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.config.nodes_in[i], 1)) for i in blocknums]
        )
        self.layer_pos_linears = nn.ModuleList(
            [nn.Linear(self.config.nodes_in[i]*config.input_dim, self.config.nodes_in[i]) for i in blocknums]
        )
        self.next_layer_pos_linears = nn.ModuleList(
            [nn.Linear(self.config.nodes_in[i]*config.input_dim, self.config.nodes_out[i]) for i in blocknums]
        )
        decblocks = []
        for i in blocknums:
            print(f'in steps: {config.periods_out[i]}')
            print(f'out steps: {config.periods_in[i]}')
            if i == blocknums[-1]:
                out_nodes = 128
            else:
                out_nodes = config.nodes_in[i]
            in_nodes = config.nodes_out[i]
            
            decblocks.append(
                ASTGCNBlock(
                    config.K,
                    in_nodes,
                    out_nodes,
                    config.num_labels,
                    config.input_dim,
                    config.input_dim,
                    config.time_kernels[i],
                    1,
                    config.periods_out[i],
                    config.periods_in[i],
                    config.input_dim,
                    config.input_dim,
                    config.num_nodes,
                    config.input_dim,
                    config.time_kernels[i],
                    temporal_conv=True,
                    causal_mask=True
                )
            )

        self.decoder = nn.ModuleList(
            decblocks
        )

        min_seq = min(config.periods_out)
        min_nodes = min(config.nodes_out + config.nodes_in)
        self.composer_pooling = ASTGCNBlock(
                    config.K,
                    min_nodes,
                    1,
                    config.num_labels,
                    config.input_dim,
                    config.input_dim,
                    config.time_kernels[i],
                    1,
                    min_seq,
                    min_seq,
                    config.input_dim,
                    config.input_dim,
                    config.num_nodes,
                    config.input_dim,
                    config.time_kernels[i],
                    temporal_conv=True,
                    causal_mask=True
                )

        self._final_conv = nn.Conv2d(
            config.periods,
            config.periods,
            kernel_size=(1, config.input_dim),
            bias=False
        ) 

        self.pooled_edge_indices = {}
        
        self.latent_encoder = nn.Linear(min_seq*config.num_nodes*config.input_dim, config.hidden_dim*2, bias=True)
        self.latent_decoder = nn.Linear(config.hidden_dim, min_seq*config.num_nodes*config.input_dim, bias=True)
        self.composer_hidden = nn.Linear(min_seq*config.input_dim, config.hidden_dim, bias=False)
        self.composer_output = nn.Linear(config.hidden_dim, config.num_composers, bias=False)

        self.config = config


    @torch.no_grad()
    def make_positional_sinusoids(self):
        """
        Create and register a positional sinusoid buffer for space and time encoding.
        """
        # Dimensions
        half_input_dim = self.input_dim // 2
        num_labels = self.num_labels
        num_periods = self.periods
        above_which_are_zeros = 128 + self.config.max_voices + self.config.max_rhythm # +1 for padding

        # Generate position values
        time_positions = torch.linspace(0, torch.pi / 2., num_labels, dtype=torch.float)
        space_positions = torch.linspace(0, torch.pi / 2., num_periods, dtype=torch.float)

        # Frequency grids
        time_freqs = torch.logspace(0, 1, half_input_dim * num_labels, num_periods).view(num_labels, half_input_dim)
        space_freqs = torch.logspace(0, 1, half_input_dim * num_periods, num_labels).view(num_periods, half_input_dim)

        # Scale positions by frequencies
        scaled_time = time_freqs.reshape(-1, self.num_labels) * time_positions  # [num_labels, half_input_dim]
        scaled_space = space_freqs.reshape(-1, self.periods) * space_positions  # [num_periods, half_input_dim]

        # Prepare indices for even/odd feature dimensions
        even_indices = torch.arange(0, half_input_dim * 2, 2, dtype=torch.long)
        odd_indices = torch.arange(1, half_input_dim * 2, 2, dtype=torch.long)

        # Sinusoids for time
        time_sin = torch.sin(scaled_time.t()).unsqueeze(-1).expand(num_labels, half_input_dim, num_periods)
        time_cos = torch.cos(scaled_time.t()).unsqueeze(-1).expand(num_labels, half_input_dim, num_periods)

        # Sinusoids for space
        space_sin = torch.sin(scaled_space).unsqueeze(0).expand(num_labels, half_input_dim, num_periods)
        space_cos = torch.cos(scaled_space).unsqueeze(0).expand(num_labels, half_input_dim, num_periods)

        # Allocate and fill space sinusoid tensor
        space_sinusoids = torch.zeros(num_labels, half_input_dim * 2, num_periods)
        space_sinusoids[:, even_indices, :] = space_sin
        space_sinusoids[:, odd_indices, :] = space_cos
        space_sinusoids = space_sinusoids.unsqueeze(0)  # [1, num_labels, input_dim, periods]

        # Allocate and fill time sinusoid tensor
        time_sinusoids = torch.zeros(num_labels, half_input_dim * 2, num_periods)
        time_sinusoids[:, even_indices, :] = time_sin
        time_sinusoids[:, odd_indices, :] = time_cos
        time_sinusoids = time_sinusoids.unsqueeze(0)  # [1, num_labels, input_dim, periods]

        # Concatenate space and time sinusoids along feature dimension
        positional_sinusoids = torch.cat([space_sinusoids, time_sinusoids], dim=2)  # [1, num_labels, input_dim*2, periods]

        # Optionally zero out the first label (e.g., for padding)       
        positional_sinusoids[:, 0, :, :] = 0.0
        positional_sinusoids[:, above_which_are_zeros:, :, :] = 0.0 # zero out composer
        # Register as buffer for use in the model
        self.register_buffer('positional_sinusoids', positional_sinusoids)

    def sparse_embedding_lookup(self, indices, embedding_layer):
        """
        Args:
            indices: tensor of shape [batch, 233, seq_len] with values in range [0, 233]
            embedding_layer: nn.Embedding layer
        
        Returns:
            tensor of shape [batch, 233, embedding_dim, seq_len] where embeddings
            appear at positions corresponding to their indices
        """
        batch_size, num_positions, seq_len = indices.shape
        embedding_dim = embedding_layer.embedding_dim
        
        # Create output tensor filled with zeros
        output = torch.zeros(batch_size, num_positions, embedding_dim, seq_len, 
                            device=indices.device, dtype=embedding_layer.weight.dtype,requires_grad=False)
        
        # Find non-zero positions
        nonzero_mask = indices != 0  # assuming 0 is your padding/empty index
        
        if nonzero_mask.any():
            # Get batch, position, and time indices where we have non-zero values
            batch_idx, pos_idx, time_idx = torch.where(nonzero_mask)
            
            # Get the actual index values at those positions
            actual_indices = indices[batch_idx, pos_idx, time_idx].clone()
            
            # Get embeddings for these indices
            embeddings = embedding_layer(actual_indices.clone())  # shape: [num_nonzero, embedding_dim]
            actual_indices[actual_indices > num_positions - 1] = num_positions-1
            # Place embeddings at their corresponding positions
            output[batch_idx, actual_indices, :, time_idx] = embeddings
        
        return output
    
    def get_inputs(self, x):
        batch_size = x.shape[0]
        x_indexes = x.to(torch.long)
        x_embedding = self.sparse_embedding_lookup(x_indexes, self.src_embedding)
        x_selected = torch.gather(self.positional_sinusoids.expand(batch_size, -1,-1,-1),1, x_indexes.unsqueeze(-2).expand(-1, -1, self.input_dim, -1).to(torch.long))
        return x_embedding + x_selected

    def forward(self, databatch):
        """
        Forward pass using simplified graph structure.

        Args:
            databatch: Dictionary containing:
                - 'features': Input features tensor
                - 'feature_indices': Feature indices for embedding lookup
                - 'input_graphs': List of batched graphs per timestep
                - 'target_graphs': List of batched target graphs per timestep
                - 'global_graph': Union of all timesteps' graphs (optional, used if config.use_global_graph=True)
        """
        x = databatch['features']
        x_indices = databatch['feature_indices']
        x_emb = self.get_inputs(x_indices)
        x = x.unsqueeze(-2) * x_emb

        use_global = getattr(self.config, 'use_global_graph', False)

        use_global = True  # TODO: make configurable
        if use_global:
            # Use global graph - just pass the single graph, blocks will reuse it
            global_graph = databatch['global_graph']
            input_graphs = global_graph  # Single graph, not a list
            target_graphs = global_graph
        else:
            # Use per-timestep graphs
            input_graphs = databatch['input_graphs']
            target_graphs = databatch['target_graphs']

        # Encoder pass
        current_graphs = input_graphs
        for i in range(self.config.num_blocks):
            x, _, current_graphs, _ = self.encoder[i](x, current_graphs, current_graphs, pool_first=False)
            x = F.dropout(x, GLOBAL_DROPOUT, self.training)

        # Decoder pass
        current_graphs = target_graphs if self.training else current_graphs
        for i in range(self.config.num_blocks):
            x, _, current_graphs, _ = self.decoder[i](x, current_graphs, current_graphs, pool_first=False)

        # Final convolution and output
        x = self._final_conv(x[:,:-1, :, :].permute(0, 3, 1, 2))
        x = x[:, :, :, -1]
        x = x.permute(0, 2, 1)
        x = F.log_softmax(x, dim=-2)
        return x

    def predict(self, databatch):
        x, _, _ = self.forward(databatch)
        return x.argmax(dim=1)