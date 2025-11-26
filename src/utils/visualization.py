import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from pathlib import Path
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

# Import threshold from metrics module
from src.training.metrics import NOTE_THRESHOLD


def visualize_predictions(inputs, targets, filenames, metrics_list=None, save_path=None, max_samples=10):
    """
    Visualize batch predictions with input/target comparison.

    Args:
        inputs: Tensor of shape [batch, node, seq] or [batch, seq] - model predictions (probabilities, not log probs)
        targets: Tensor of shape [batch, node, seq] or [batch, seq] - ground truth
        filenames: List of length batch - file identifiers for each example
        metrics_list: Optional list of metric dicts (one per sample) to display. If None, no metrics shown.
        save_path: Optional path to save the visualization
        max_samples: Maximum number of samples to visualize (default 10)

    Returns:
        None
    """
    if torch.is_tensor(inputs):
        inputs = inputs.detach().cpu()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu()

    # Randomly sample max_samples from the batch
    original_batch_size = inputs.shape[0]
    if original_batch_size > max_samples:
        indices = torch.randperm(original_batch_size)[:max_samples]
        inputs = inputs[indices]
        targets = targets[indices]
        filenames = [filenames[i] for i in indices.tolist()]

    # Handle shape mismatches by reshaping to [batch, -1]
    if inputs.shape != targets.shape:
        print(f"Warning: Shape mismatch - inputs: {inputs.shape}, targets: {targets.shape}")
        # Flatten to 2D and match sizes
        batch_size = min(inputs.shape[0], targets.shape[0])
        inputs = inputs[:batch_size].reshape(batch_size, -1)
        targets = targets[:batch_size].reshape(batch_size, -1)

        # Match the second dimension
        min_dim = min(inputs.shape[1], targets.shape[1])
        inputs = inputs[:, :min_dim]
        targets = targets[:, :min_dim]
    else:
        batch_size = inputs.shape[0]

    # Ensure we have at least 2D tensors
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)

    # Reshape to 3D if needed [batch, nodes, seq]
    if inputs.ndim == 2:
        # Assume [batch, features] - reshape to [batch, nodes, seq]
        # Try to make a reasonable square-ish shape
        total_features = inputs.shape[1]
        nodes = int(np.sqrt(total_features))
        seq = total_features // nodes
        if nodes * seq < total_features:
            seq += 1
        inputs = torch.nn.functional.pad(inputs, (0, nodes * seq - total_features))
        inputs = inputs.reshape(batch_size, nodes, seq)
        targets = torch.nn.functional.pad(targets, (0, nodes * seq - total_features))
        targets = targets.reshape(batch_size, nodes, seq)

    # Create visualization
    fig, axes = plt.subplots(batch_size, 1, figsize=(16, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        # Create a combined visualization showing target, prediction, and difference
        target_roll = targets[i].numpy()  # [node, seq]
        input_roll = inputs[i].numpy()    # [node, seq]

        # Remove nodes that are inactive for the entire sequence
        # A node is active if it has any value > threshold in either target or input
        active_mask = (target_roll.max(axis=1) > NOTE_THRESHOLD) | (input_roll.max(axis=1) > NOTE_THRESHOLD)

        if active_mask.sum() > 0:
            target_roll = target_roll[active_mask]
            input_roll = input_roll[active_mask]
        # else: keep all nodes if none are active

        # Create RGB image where:
        # - Green = correct predictions (TP)
        # - Red = false positives (predicted but not in target)
        # - Blue = false negatives (in target but not predicted)
        # - Black = true negatives (silence)

        target_bin = (target_roll > NOTE_THRESHOLD)
        input_bin = (input_roll > NOTE_THRESHOLD)

        rgb_image = np.zeros((*target_roll.shape, 3))

        # True positives - green
        rgb_image[target_bin & input_bin] = [0, 1, 0]
        # False positives - red
        rgb_image[~target_bin & input_bin] = [1, 0, 0]
        # False negatives - blue
        rgb_image[target_bin & ~input_bin] = [0, 0, 1]

        # Don't transpose - keep as [node, seq] which displays correctly with nodes on Y-axis
        axes[i].imshow(rgb_image, aspect='auto', origin='lower', interpolation='nearest')

        # Add filename and metrics to title
        title = f"{filenames[i]}\n"
        if metrics_list and i < len(metrics_list):
            m = metrics_list[i]
            title += f"Accuracy: {m['note_accuracy']:.3f} | Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1']:.3f}"
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel("Pitch (Node)")

        # Add y-axis with note names at octave boundaries (less frequent, more readable)
        # Show labels every 24 nodes (2 octaves) instead of every 12
        num_nodes = target_roll.shape[0]
        tick_step = max(24, num_nodes // 10)  # At most 10 ticks
        tick_positions = np.arange(0, num_nodes, tick_step)
        axes[i].set_yticks(tick_positions)
        axes[i].set_yticklabels([f"{p}" for p in tick_positions], fontsize=8)

        # X-axis ticks - show every ~10% of sequence
        num_seq = target_roll.shape[1]
        x_tick_step = max(1, num_seq // 10)
        x_tick_positions = np.arange(0, num_seq, x_tick_step)
        axes[i].set_xticks(x_tick_positions)
        axes[i].set_xticklabels([f"{p}" for p in x_tick_positions], fontsize=8)

        # Remove minor ticks
        axes[i].tick_params(which='both', length=0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct (TP)'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative'),
        Patch(facecolor='black', label='True Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def visualize_activations(activations_dict, inputs_dict, num_layers, model_input=None, model_output=None, node_mask=None, save_path=None):
    """
    Visualize all intermediate activations from model layers in a single figure.
    Picks one random example from the batch.
    Shows: input -> encoder layers -> decoder layers -> output
    For hidden layers, shows node-level energy (mean of squared features).

    Args:
        activations_dict: Dictionary with keys like 'encoderlayer0', 'decoderlayer0', etc.
                         containing activation tensors
        inputs_dict: Dictionary with keys like 'encoderlayer0', 'decoderlayer0', etc.
                    containing input tensors (not used, kept for compatibility)
        num_layers: Number of layers to visualize
        model_input: Original input to the model (shape: batch, nodes, features, time)
        model_output: Final output from the model after exp() (shape: batch, nodes, time)
        node_mask: Optional boolean mask (shape: batch, nodes) indicating active nodes in the graph
        save_path: Optional path to save the visualization
    """
    # Pick one random example index (consistent across all layers)
    random_idx = None

    # Collect all layer data
    all_layer_data = []

    # Add model input at the top if provided
    if model_input is not None:
        all_layer_data.append(('Input', model_input, True))  # True = is input/output

    # Add all encoder layers first
    for layer in range(num_layers):
        encoder_act = activations_dict.get(f'encoderlayer{layer}')
        if encoder_act is not None:
            all_layer_data.append((f'Encoder {layer}', encoder_act, False))

    # Then add all decoder layers
    for layer in range(num_layers):
        decoder_act = activations_dict.get(f'decoderlayer{layer}')
        if decoder_act is not None:
            all_layer_data.append((f'Decoder {layer}', decoder_act, False))

    # Add model output at the bottom if provided
    if model_output is not None:
        all_layer_data.append(('Output', model_output, True))  # True = is input/output

    num_plots = len(all_layer_data)

    if num_plots == 0:
        print("No activation data to visualize")
        return

    # Pick random example index first (before computing active nodes mask)
    first_data = all_layer_data[0][1]
    if torch.is_tensor(first_data):
        first_data = first_data.detach().cpu()
    if first_data.ndim >= 1:
        batch_size = first_data.shape[0]
        random_idx = torch.randint(0, batch_size, (1,)).item()
    else:
        random_idx = 0

    # Get the node mask for the selected example if provided
    if node_mask is not None:
        if torch.is_tensor(node_mask):
            node_mask = node_mask.detach().cpu()
        active_nodes_mask = node_mask[random_idx].numpy()  # [num_nodes]
    else:
        active_nodes_mask = None

    # Process each layer to get 2D representations
    # Store the 2D data for each layer to avoid recomputing
    layer_data_2d = []

    for name, data, is_input_output in all_layer_data:
        if torch.is_tensor(data):
            data = data.detach().cpu()

        # Select the random example
        if data.ndim >= 1:
            data = data[random_idx]

        # Convert to 2D [nodes, time]
        if is_input_output:
            if data.ndim == 3:
                data_2d = data.mean(dim=1)  # [nodes, features, time] -> [nodes, time]
            elif data.ndim == 2:
                data_2d = data
            else:
                data_2d = data.unsqueeze(0)
        else:
            if data.ndim == 3:
                data_2d = (data ** 2).mean(dim=1)  # [nodes, features, time] -> [nodes, time]
            elif data.ndim == 2:
                data_2d = data
            else:
                data_2d = data.unsqueeze(0)

        layer_data_2d.append(data_2d)

    # Create vertical stack of subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 2 * num_plots))
    if num_plots == 1:
        axes = [axes]

    for idx, (name, data, is_input_output) in enumerate(all_layer_data):
        ax = axes[idx]

        # Get the precomputed 2D data for this layer
        data_2d = layer_data_2d[idx].numpy()

        # Apply the node mask if provided (mask out inactive nodes from the graph)
        if active_nodes_mask is not None and len(active_nodes_mask) == data_2d.shape[0]:
            data_2d = data_2d[active_nodes_mask]

        # Plot without any labels or ticks
        ax.imshow(data_2d, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')

        # Remove all ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Turn off axis
        ax.axis('off')

    plt.subplots_adjust(hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def visualize_graph_structure(data, save_path=None, render_3d=True, max_samples=10):
    """
    Visualize graph structure from a PyG Data or Batch object.
    Renders the graph in 3D (optional) with node labels and edge weights,
    and displays the dense adjacency matrix for active nodes.

    For Batch objects, samples a random subset of graphs.

    Args:
        data: PyG Data or Batch object with attributes:
              - edge_index: [2, num_edges] edge connectivity
              - edge_attr: [num_edges, *] optional edge weights/features
              - x: [num_nodes, num_features] node features
              - batch: [num_nodes] batch assignment (if Batch object)
        save_path: Optional path to save the visualization
        render_3d: If True, render graph in 3D; otherwise 2D
        max_samples: Maximum number of graphs to visualize from batch (default 10)
    """
    from torch_geometric.data import Batch

    # Handle both Data and Batch objects
    is_batch = isinstance(data, Batch)

    # Check for required attributes
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        print("Warning: Data object has no edge_index attribute")
        return

    # Sample random subset if batch
    if is_batch and data.num_graphs > max_samples:
        # Get random graph indices
        graph_indices = torch.randperm(data.num_graphs)[:max_samples]

        # Create mask for nodes belonging to selected graphs
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        for graph_idx in graph_indices:
            node_mask |= (data.batch == graph_idx)

        # Filter edges to only include those with both nodes in selected graphs
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]

        # Create mapping from old node indices to new node indices
        old_to_new = torch.zeros(data.num_nodes, dtype=torch.long)
        old_to_new[node_mask] = torch.arange(node_mask.sum())

        # Extract subgraph
        edge_index = old_to_new[data.edge_index[:, edge_mask]]
        node_features = data.x[node_mask] if hasattr(data, 'x') and data.x is not None else None
        edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    else:
        # Use entire graph
        edge_index = data.edge_index
        node_features = data.x if hasattr(data, 'x') and data.x is not None else None
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

    # Identify active nodes (nodes with edges or non-zero features)
    active_from_edges = torch.unique(edge_index.flatten()) if edge_index.numel() > 0 else torch.tensor([])

    if node_features is not None:
        active_from_features = torch.where(node_features.sum(dim=-1) > 0)[0]
        active_nodes = torch.unique(torch.cat([active_from_edges, active_from_features]))
    else:
        active_nodes = active_from_edges

    if len(active_nodes) == 0:
        print("No active nodes found in the graph")
        return

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(active_nodes.cpu().numpy())

    # Add edges with weights
    if edge_index.numel() > 0:
        edges = edge_index.t().cpu().numpy()
        if edge_attr is not None:
            # Use first dimension of edge_attr as weight
            weights = edge_attr.cpu().numpy()
            if weights.ndim > 1:
                weights = weights[:, 0]
            edge_list = [(int(e[0]), int(e[1]), float(w)) for e, w in zip(edges, weights)]
            G.add_weighted_edges_from(edge_list)
        else:
            G.add_edges_from([(int(e[0]), int(e[1])) for e in edges])

    # Create figure with subplots: graph (left) and adjacency matrix (right)
    fig, (ax_graph, ax_matrix) = plt.subplots(1, 2, figsize=(18, 8))

    # === Graph Visualization (2D only) ===
    # 2D spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes without labels
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightblue',
                          node_size=100, edgecolors='navy', linewidths=1.5)

    # Draw edges with weights visually represented by width and color
    if edge_attr is not None:
        edge_weights = np.array([G[u][v].get('weight', 1.0) for u, v in G.edges()])
        # Normalize weights for visualization
        if edge_weights.max() > 0:
            edge_weights_norm = edge_weights / edge_weights.max()
        else:
            edge_weights_norm = edge_weights

        # Draw edges with varying width and alpha based on weight
        for (u, v), weight, norm_weight in zip(G.edges(), edge_weights, edge_weights_norm):
            ax_graph.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                         'k-', linewidth=norm_weight * 3 + 0.5, alpha=norm_weight * 0.8 + 0.2)
    else:
        # Draw edges with uniform width
        nx.draw_networkx_edges(G, pos, ax=ax_graph, width=1.0, alpha=0.5)

    title = "Graph Structure"
    if is_batch:
        title += f" ({len(G.nodes())} nodes)"
    ax_graph.set_title(title, fontsize=14)
    ax_graph.axis('off')

    # === Adjacency Matrix Visualization ===
    # Build dense adjacency matrix for active nodes only
    active_node_list = sorted(active_nodes.cpu().numpy())
    n_active = len(active_node_list)
    node_to_idx = {node: i for i, node in enumerate(active_node_list)}

    adj_matrix = np.zeros((n_active, n_active))

    for edge in G.edges(data=True):
        u, v = edge[0], edge[1]
        if u in node_to_idx and v in node_to_idx:
            weight = edge[2].get('weight', 1.0)
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Symmetric for undirected graph

    # Plot adjacency matrix without colorbar
    im = ax_matrix.imshow(adj_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    matrix_title = "Adjacency Matrix"
    if is_batch:
        matrix_title += f" ({n_active} nodes)"
    ax_matrix.set_title(matrix_title, fontsize=14)

    # Remove all ticks and labels
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    ax_matrix.set_xlabel("")
    ax_matrix.set_ylabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig
