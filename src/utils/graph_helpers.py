from torch_geometric.data import Data, Batch
from torch_geometric.utils import coalesce, mask_to_index, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import max_pool
import torch

import torch.nn.functional as F

def interpolate_adjacency(adj, m, mode='nearest'):
    # adj: [batch, n, n]
    # Add a channel dimension: [batch, 1, n, n]
    adj = adj.unsqueeze(1)
    # Interpolate to [batch, 1, m, m]
    adj_interp = F.interpolate(adj, size=(m, m), mode=mode)
    # Remove channel dimension: [batch, m, m]
    return adj_interp.squeeze(1)



def max_pool_batch(batch, time_slice, cluster_idxs):
    
    input_list = []
    target_list = []
    

    for batch_id, item in enumerate(batch):
        
        in_feat = torch.stack([fi for fi in item['features'][time_slice]], dim=-1).sum(dim=-1, keepdim=True).detach()
        target_feat = torch.stack([fi for fi in item['targets'][time_slice]], dim=-1).sum(dim=-1, keepdim=True).detach()
        num_nodes_in = in_feat.shape[0]
        num_nodes_out = len(cluster_idxs.unique()) if cluster_idxs is not None else num_nodes_in
        in_edge_index =  torch.cat([ei for ei in  item['input_edge_index'][time_slice]], dim=-1).detach()
        in_edge_attr = torch.cat([ei for ei in  item['input_edge_attr'][time_slice]], dim=-1).detach()
        target_edge_index =  torch.cat([ei for ei in  item['target_edge_index'][time_slice]], dim=-1).detach()
        target_edge_attr = torch.cat([ei for ei in  item['target_edge_attr'][time_slice]], dim=-1).detach()
        in_edge_index, in_edge_attr = coalesce(in_edge_index, in_edge_attr, num_nodes=num_nodes_in, reduce='sum')
        target_edge_index, target_edge_attr = coalesce(target_edge_index, target_edge_attr, num_nodes=num_nodes_in, reduce='sum')
        idata = Data(x=in_feat, edge_index=in_edge_index, edge_attr=in_edge_attr.to(torch.float), orig_batch=batch_id, num_nodes=num_nodes_in)
        tdata = Data(x=target_feat, edge_index=target_edge_index, edge_attr=target_edge_attr.to(torch.float), orig_batch=batch_id, num_nodes=num_nodes_in)
        '''if cluster_idxs is not None:
            idata = max_pool(cluster_idxs[:num_nodes_in], idata)
            tdata = max_pool(cluster_idxs[:num_nodes_in], tdata)
            idata.x = torch.unique(idata.edge_index).to(torch.bool).to(torch.float).unsqueeze(-1)
            tdata.x = torch.unique(tdata.edge_index).to(torch.bool).to(torch.float).unsqueeze(-1)
            idata.orig_batch = batch_id
            idata.num_nodes = idata.edge_index.max().item() + 1 if idata.edge_index.numel() > 0 else 0
            tdata.num_nodes = tdata.edge_index.max().item() + 1 if tdata.edge_index.numel() > 0 else 0
            tdata.orig_batch = batch_id'''
            
        input_list.append(idata)
        target_list.append(tdata)
    inputs = Batch.from_data_list(input_list, follow_batch=['orig_batch'])
    targets = Batch.from_data_list(target_list, follow_batch=['orig_batch'])

    return inputs, targets


def pool_nodes_single_slice(batch,  num_nodes_in, num_nodes_out, time_slice):

    input_list = []
    target_list = []

    for batch_id, item in enumerate(batch):
        #in_feat = torch.cat([fi for fi in item['features'][time_slice]], dim=-1).detach()

        in_edge_index =  torch.cat([ei for ei in  item['input_edge_index'][time_slice]], dim=-1).detach()
        in_edge_attr = torch.cat([ei for ei in  item['input_edge_attr'][time_slice]], dim=-1).detach()
        target_edge_index =  torch.cat([ei for ei in  item['target_edge_index'][time_slice]], dim=-1).detach()
        target_edge_attr = torch.cat([ei for ei in  item['target_edge_attr'][time_slice]], dim=-1).detach()
        in_edge_index, in_edge_attr = coalesce(in_edge_index, in_edge_attr, num_nodes=num_nodes_in, reduce='sum')
        target_edge_index, target_edge_attr = coalesce(target_edge_index, target_edge_attr, num_nodes=num_nodes_in, reduce='sum')
        in_dense_adj = to_dense_adj(in_edge_index, batch=None, edge_attr=in_edge_attr, max_num_nodes=num_nodes_in)
        in_dense_adj = interpolate_adjacency(in_dense_adj, num_nodes_out, mode='nearest')
        in_mask = torch.abs(in_dense_adj.sum(dim=-1)) > 0

        target_dense_adj = to_dense_adj(target_edge_index, batch=None, edge_attr=target_edge_attr, max_num_nodes=num_nodes_in)
        target_dense_adj = interpolate_adjacency(target_dense_adj, num_nodes_out, mode='nearest')
        target_mask = torch.abs(target_dense_adj.sum(dim=-1)) > 0

        iedge_index, iedge_attr = dense_to_sparse(in_dense_adj, mask=in_mask)
        tedge_index, tedge_attr = dense_to_sparse(target_dense_adj, mask=target_mask)
        
        if iedge_index.numel() > 0:
            iedge_index[0,:] += mask_to_index(in_mask.squeeze()).min()
        if tedge_index.numel() > 0:
            tedge_index[0,:] += mask_to_index(target_mask.squeeze()).min()
        input_list.append(Data(edge_index=iedge_index,
                                 edge_attr=iedge_attr.to(torch.float),
                                 num_nodes=num_nodes_out,
                                 x_mask=in_mask,
                                 orig_batch=batch_id))
        target_list.append(Data(edge_index=tedge_index,
                                 edge_attr=tedge_attr.to(torch.float),
                                 num_nodes=num_nodes_out,
                                 x_mask=target_mask,
                                 orig_batch=batch_id))
        
    inputs = Batch.from_data_list(input_list, follow_batch=['x_mask', 'orig_batch'])
    targets = Batch.from_data_list(target_list, follow_batch=['x_mask', 'orig_batch'])


    return inputs, targets



def pool_edge_indices(datalist, output_length=10, reduce='mean'):
    """
    Pools a list of edge_index tensors (and optionally edge_attr tensors) into m pooled edge_index tensors.
    Each pooled edge_index is the coalesced union of a chunk of the input list.
    Args:
        edge_index_list: list of [2, num_edges_i] tensors
        edge_attr_list: list of [num_edges_i] tensors or None
        m: number of pooled outputs
        reduce: 'mean', 'sum', etc. (see torch_scatter)
    Returns:
        pooled_edge_indices: list of [2, pooled_num_edges_j] tensors
        pooled_edge_attr: list of [pooled_num_edges_j] tensors (if edge_attr_list is not None)
    """
    n = len(datalist)
    chunk_size = n // output_length
    pooled_edge_indices = []
    pooled_edge_attr = []
    pooled_xs = []
    outDataList = []
    for i in range(output_length):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < output_length - 1 else n
        chunk = datalist[start:end]
        if chunk[0].edge_attr is not None:
            chunk_weights = [c.edge_attr for c in chunk if c.edge_attr is not None]
            chunk_edge_indices = [c.edge_index for c in chunk if c.edge_index is not None]
            # Concatenate all edge indices and weights in the chunk
            edge_index_cat = torch.cat(chunk_edge_indices, dim=1)
            edge_attr_cat = torch.cat(chunk_weights)
            # Coalesce (deduplicate) edges
            edge_index_co, edge_attr_co = coalesce(edge_index_cat, edge_attr_cat, reduce=reduce)
            pooled_edge_indices.append(edge_index_co)
            pooled_edge_attr.append(edge_attr_co)
        else:
            edge_index_cat = torch.cat(chunk, dim=1)
            edge_index_co, _ = coalesce(edge_index_cat, None)
            pooled_edge_indices.append(edge_index_co)
        pooled_xs.append(torch.stack([c.x for c in chunk if c.x is not None], dim=1).sum(dim=1)>0.0)


        
    outDataList = [Data(edge_index=edge_index,
                                 edge_attr=edge_attr.to(torch.float), x=pooled_x) for (edge_index, edge_attr, pooled_x) in zip(pooled_edge_indices, pooled_edge_attr, pooled_xs)]

    return outDataList


def unpool_graphs(graph_list, target_length):
    """
    Repeat/interpolate graphs to match target length.

    Args:
        graph_list: List of PyG Batch objects, Data objects, or single temporal batch
        target_length: Desired output length

    Returns:
        List of PyG Batch/Data objects with length = target_length
    """
    # Handle temporal batch (single Data object with time_index)
    if len(graph_list) == 1 and hasattr(graph_list[0], 'time_index'):
        temporal_batch = graph_list[0]
        current_length = temporal_batch.num_timesteps

        if current_length == target_length:
            return graph_list

        # Remap time indices via repetition
        repeat_factor = target_length // current_length
        remainder = target_length % current_length

        # Create new time index mapping
        new_time_indices = []
        for old_t in range(current_length):
            repeats = repeat_factor + (1 if old_t < remainder else 0)
            new_time_indices.extend([old_t] * repeats)

        # Map old time indices to new time indices
        time_mapping = torch.zeros(current_length, dtype=torch.long)
        new_time_counter = 0
        for old_t in range(current_length):
            time_mapping[old_t] = new_time_counter
            repeats = repeat_factor + (1 if old_t < remainder else 0)
            new_time_counter += repeats

        # Update time indices in the batch
        old_time_idx = temporal_batch.time_index
        new_time_idx = time_mapping[old_time_idx]

        # Create new temporal batch with updated indices
        unpooled_batch = Data(
            edge_index=temporal_batch.edge_index.clone(),
            edge_attr=temporal_batch.edge_attr.clone() if temporal_batch.edge_attr is not None else None,
            time_index=new_time_idx,
            batch_index=temporal_batch.batch_index.clone(),
            num_nodes=temporal_batch.num_nodes,
            num_timesteps=target_length,
            batch_size=temporal_batch.batch_size
        )

        return [unpooled_batch]

    # Handle list of regular graphs (original behavior)
    if len(graph_list) == target_length:
        return graph_list

    repeat_factor = target_length // len(graph_list)
    remainder = target_length % len(graph_list)

    unpooled = []
    for i, graph in enumerate(graph_list):
        repeats = repeat_factor + (1 if i < remainder else 0)
        unpooled.extend([graph] * repeats)

    return unpooled


def pool_temporal_batch(temporal_batch, target_length, reduce='mean'):
    """
    Pool a temporal batch by merging consecutive timesteps.

    Args:
        temporal_batch: Data object with time_index attribute
        target_length: Target number of timesteps
        reduce: How to combine edge attributes ('mean', 'max', 'sum')

    Returns:
        New Data object with pooled timesteps
    """
    if not hasattr(temporal_batch, 'time_index'):
        raise ValueError("Expected temporal batch with time_index attribute")

    current_length = temporal_batch.num_timesteps

    if current_length == target_length:
        return temporal_batch

    # Compute chunk mapping: which new timestep does each old timestep map to?
    chunk_size = current_length / target_length
    time_mapping = torch.floor(temporal_batch.time_index / chunk_size).long()
    time_mapping = torch.clamp(time_mapping, 0, target_length - 1)

    # Update temporal batch with new time indices
    pooled_batch = Data(
        edge_index=temporal_batch.edge_index.clone(),
        edge_attr=temporal_batch.edge_attr.clone() if temporal_batch.edge_attr is not None else None,
        time_index=time_mapping,
        batch_index=temporal_batch.batch_index.clone(),
        num_nodes=temporal_batch.num_nodes,
        num_timesteps=target_length,
        batch_size=temporal_batch.batch_size
    )

    return pooled_batch