import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import coalesce
from torch_geometric.data import Data, Batch
from pathlib import Path


def batch_to_device(batch, device):
    """
    Move all tensors in a batch dictionary to the specified device.
    Handles nested structures including lists of PyG Batch objects.

    Args:
        batch: Dictionary containing tensors and PyG objects
        device: Target device (e.g., torch.device('mps'), 'cuda', 'cpu')

    Returns:
        Dictionary with all tensors moved to device
    """
    if batch is None:
        return None

    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, Batch):
            result[key] = value.to(device)
        elif isinstance(value, list):
            # Handle lists of Batch objects (input_graphs, target_graphs)
            if len(value) > 0 and isinstance(value[0], Batch):
                result[key] = [v.to(device) for v in value]
            elif len(value) > 0 and isinstance(value[0], torch.Tensor):
                result[key] = [v.to(device) for v in value]
            else:
                # Keep non-tensor lists as-is (e.g., composer_names, file_paths)
                result[key] = value
        else:
            result[key] = value

    return result

class MidiGraphDataset(Dataset):
    def __init__(self, npz_dir, seq_length=32, time_step=0.25, max_pitch=127, config=None):
        """
        A simplified MIDI dataset using a sequence of PyG Data objects
        
        Args:
            npz_dir: Directory with preprocessed sparse MIDI files
            seq_length: Number of time steps to include
            time_step: Time between steps in seconds
            max_pitch: Maximum MIDI pitch to consider
        """
        self.npz_dir = Path(npz_dir)
        self.seq_length = seq_length
        self.time_step = time_step
        self.max_pitch = max_pitch
        self.lags = 20
        self.config = config
        print(f"Loading dataset from {self.npz_dir}")
        print(f'loading composer mapping')
        # Load composer mapping

        # Build segment index
        self.segments = self._build_segment_index()
        
        # Fixed feature dimensions for consistency

    def _build_segment_index(self):
        """Create an index of valid segments across all files"""
        segments = []
        
        print("Building segment index...")
        for path in self.npz_dir.glob('*/*.npz'):
            # Skip metadata files
            if path.parent.name == "processing_stats.json" or path.name == "composer_map.json":
                continue
                
            try:
                # Just load metadata without loading all events
                data = np.load(path, allow_pickle=True)
                events = data['events']
                if len(events) == 0:
                    continue
                    
                end_time = np.max(events[:, 1])  # Max end time
                
                # Create segments with 50% overlap
                segment_duration = self.seq_length * self.time_step
                
                for start_time in np.arange(0, end_time - segment_duration + 0.01, 40):
                    segments.append((str(path), start_time))
            except Exception as e:
                print(f"Error indexing {path}: {e}")
                continue
        
        print(f"Found {len(segments)} segments across {len(set(s[0] for s in segments))} files")
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Returns a single PyG Data object containing the full sequence
        """
        # Get segment information
        file_path, start_time = self.segments[idx]
        
        # Load the requested file
        data = np.load(file_path, allow_pickle=True)
        events = data['events']
        composer_id = int(data['composer_id'])
        composer_name = self.get_composer_name(composer_id)
        composer_node_id = self.config.num_nodes - 1 if self.config else 0
        
        # Initialize lists to store all time steps
        all_time_steps = []
        
        for t in range(self.seq_length+self.lags):
            current_time = start_time + (t * self.time_step)
            
            # Find active notes at this time step
            active_notes = events[
                (events[:, 0] <= current_time) &  # Start time <= current time
                (events[:, 1] >= current_time)    # End time >= current time
            ]
            
            # Store pitches and voices active at this time step
            active_pitches = []
            active_velocities = []
            active_voices = []
            active_beat_ratios = []
            edge_index = []
            src_nodes = []
            dest_nodes = []
            edge_attr = []

            for note in active_notes:
                pitch = int(note[2])
                velocity = note[3] / 127.0  # Normalize to [0,1]
                voice_id = int(note[5]) + self.config.num_pitches
                beat_ratio = self.get_rhythm_label(note[6]) + self.config.num_voices + self.config.num_pitches if len(note) > 6 else 0
                if beat_ratio > self.config.num_nodes-1:
                    print(f'beat_ratio is fucked: {beat_ratio}')
                elif voice_id > self.config.num_nodes-1:
                    print(f'voice_id is fucked: {voice_id}')
                elif pitch > self.config.num_nodes-1:
                    print(f'pitch is fucked: {pitch}')
                if 0 <= pitch < self.config.num_pitches:
                    active_pitches.append(pitch)
                    active_velocities.append(velocity)
                if self.config.num_pitches <= voice_id < self.config.num_voices + self.config.num_pitches:
                    active_voices.append(voice_id)
                    active_velocities.append(velocity)
                if self.config.max_voices + self.config.max_pitch <= beat_ratio < self.config.max_voices + self.config.max_pitch + self.config.max_rhythm:
                    active_beat_ratios.append(beat_ratio)
                    active_velocities.append(velocity)
                

                # pitch x voice
                src_nodes.append(pitch)
                dest_nodes.append(voice_id)
                edge_attr.append(velocity)

                # voice x pitch
                dest_nodes.append(voice_id)
                src_nodes.append(pitch)
                edge_attr.append(velocity)

                # voice x beat_ratio
                src_nodes.append(voice_id)
                dest_nodes.append(beat_ratio)
                edge_attr.append(velocity)

                # beat_ratio x voice
                src_nodes.append(beat_ratio)
                dest_nodes.append(voice_id)
                edge_attr.append(velocity)

                # pitch x composer
                src_nodes.append(pitch)
                dest_nodes.append(composer_node_id) # last node is composer, which will be populated by different embeddings
                edge_attr.append(velocity)

                # composer x pitch
                src_nodes.append(composer_node_id)
                dest_nodes.append(pitch)
                edge_attr.append(velocity)

                # composer x voice
                src_nodes.append(composer_node_id)
                dest_nodes.append(voice_id)
                edge_attr.append(velocity)

                # voice x composer
                src_nodes.append(voice_id)
                dest_nodes.append(composer_node_id) # last node is composer, which will be populated by different embeddings
                edge_attr.append(velocity)

            # Optional: Connect all simultaneous pitches (expensive, may be redundant)
            if self.config.connect_simultaneous_pitches:
                for (src_pitch, src_veloc) in zip(active_pitches, active_velocities):
                    for (dest_pitch, dest_veloc) in zip(active_pitches, active_velocities):
                        src_nodes.append(src_pitch)
                        dest_nodes.append(dest_pitch)
                        edge_attr.append((src_veloc+dest_veloc)/2.) # edge weight between pitches is the average velocity
            src_nodes.append(composer_node_id) # make sure there's at least one edge at every timestep, even between a composer and themselves
            dest_nodes.append(composer_node_id)
            edge_attr.append(1.)
            edge_index, edge_attr = coalesce(torch.stack([torch.tensor(src_nodes, dtype=torch.long), torch.tensor(dest_nodes, dtype=torch.long)]), torch.tensor(edge_attr, dtype=torch.float32), reduce='mean')


            # Store this time step
            all_time_steps.append({
                'pitches': active_pitches,
                'velocities': active_velocities,
                'beat_ratios': active_beat_ratios,
                'voices': active_voices,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'file_path': file_path,
            })
        



        # Create piano roll tensor (time_steps × max_voices+1)
        # Values represent velocities (0 = note off)
        piano_roll = torch.zeros(self.seq_length,  self.config.num_nodes if self.config else 128)
        features = torch.zeros((self.seq_length, self.config.num_nodes))

        targets = torch.zeros((self.seq_length, self.config.num_nodes))
        feature_indices = torch.zeros((self.seq_length, self.config.num_nodes))
        target_indices = torch.zeros((self.seq_length, self.config.num_nodes))
        # Create voice roll tensor (time_steps × max_pitch+1)
        # Values represent voice IDs (-1 = no voice/note off)
        voice_roll = torch.zeros(self.seq_length, self.config.num_pitches)
        edge_attr = []

        input_edge_index = []
        input_edge_attr = []
        target_edge_index = []
        target_edge_attr = []
        file_paths = []
        

        # Fill in piano roll and voice roll
        for t, (step_data, step_targets) in enumerate(zip(all_time_steps[:-self.lags], all_time_steps[self.lags:])):
            if len(step_targets) < 1:
                print(f"Skipping empty time step {t} for segment {file_path}")
                continue
            for pitch, velocity, voice, beat_ratio in zip(
                step_data['pitches'], 
                step_data['velocities'],
                step_data['voices'],
                step_data['beat_ratios']
            ):
                features[t, pitch] = velocity
                features[t, voice] = velocity
                features[t, beat_ratio] = velocity
                feature_indices[t, pitch] = pitch
                feature_indices[t, voice] = voice
                feature_indices[t, beat_ratio] = beat_ratio
                
                #if 0 <= pitch <= self.config.num_voices+self.config.num_rhythms:

                piano_roll[t, pitch] = velocity
                piano_roll[t, voice] = velocity
                piano_roll[t, beat_ratio] = velocity
                voice_roll[t, pitch] = voice

            
            
            input_edge_index.append(step_data['edge_index'])
            input_edge_attr.append(step_data['edge_attr'])

            file_paths.append(step_data['file_path'])



            for pitch, velocity, voice, beat_ratio  in zip(
                step_targets['pitches'], 
                step_targets['velocities'],
                step_targets['voices'],
                step_targets['beat_ratios']

            ):

                targets[t, pitch] = velocity
                targets[t, voice] = velocity
                targets[t, beat_ratio] = velocity
                target_indices[t, pitch] = pitch
                target_indices[t, voice] = voice
                target_indices[t, beat_ratio] = beat_ratio
            target_edge_index.append(step_targets['edge_index'])
            target_edge_attr.append(step_targets['edge_attr'])

        max_node = self.config.num_nodes-1
        features[:, max_node] = 1.
        feature_indices[:, max_node] = composer_id + self.config.num_nodes
        targets[:, max_node] = 1.
        target_indices[:, max_node] = composer_id + self.config.num_nodes
        data = {
            'piano_roll': piano_roll,
            'voice_roll': voice_roll,
            'composer_id': composer_id,
            'composer_name': composer_name,
            'input_edge_index': input_edge_index,
            'input_edge_attr': input_edge_attr,
            'target_edge_index':target_edge_index,
            'target_edge_attr': target_edge_attr,
            'features': features,
            'targets': targets,
            'feature_indices': feature_indices,
            'target_indices': target_indices,
            'total_labels': self.config.num_labels,
            'file_paths': file_paths,
            'num_nodes': self.config.num_nodes,
            'node_mask': features.sum(dim=0) > 0,  # Mask for nodes with features
        }

        return data

    def get_composer_name(self, composer_id):
        """Get composer name from ID"""
        return self.config.get_composer_name(composer_id) if self.config else "Unknown"
    
    def get_rhythm_label(self, rhythm_value):
        rhythm_value = min(float(rhythm_value), 5.0)
        return min(int((rhythm_value / 0.125)), 38)






def temporal_graph_collate(batch, use_global_graph=True):
    """
    Collate function for batching temporal graph data.

    Args:
        batch: List of data items from dataset
        use_global_graph: If True, only build global graph (faster). If False, build per-timestep graphs.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Stack features and targets
    features = torch.stack([item['features'].transpose(0, 1) for item in batch])
    targets = torch.stack([item['targets'].transpose(0, 1) for item in batch])

    # Stack indices
    feature_indices = torch.stack([item['feature_indices'] for item in batch])
    target_indices = torch.stack([item['target_indices'] for item in batch])

    # Collect metadata
    composer_ids = [torch.tensor([item['composer_id']]) for item in batch]
    composer_names = [item['composer_name'] for item in batch]
    file_paths = [item['file_paths'][0] for item in batch]
    node_mask = torch.stack([item['node_mask'] for item in batch])

    # Conditionally create per-timestep graphs (expensive, only if needed)
    if use_global_graph:
        # Skip per-timestep graph construction - use empty lists as placeholders
        input_graphs = []
        target_graphs = []
    else:
        # Build per-timestep graphs (slower)
        seq_len = len(batch[0]['input_edge_index'])
        input_graphs = []
        target_graphs = []

        for t in range(seq_len):
            # Batch input graphs for this timestep
            input_data_list = []
            target_data_list = []

            for item in batch:
                input_data_list.append(Data(
                    edge_index=item['input_edge_index'][t],
                    edge_attr=item['input_edge_attr'][t],
                    num_nodes=item['num_nodes']
                ))
                target_data_list.append(Data(
                    edge_index=item['target_edge_index'][t],
                    edge_attr=item['target_edge_attr'][t],
                    num_nodes=item['num_nodes']
                ))

            input_graphs.append(Batch.from_data_list(input_data_list))
            target_graphs.append(Batch.from_data_list(target_data_list))

    # Build global graph per batch item (union of all timesteps)
    global_graph_list = []
    for item in batch:
        all_edge_indices = []
        all_edge_attrs = []

        # Collect edges from all input timesteps
        for t in range(len(item['input_edge_index'])):
            all_edge_indices.append(item['input_edge_index'][t])
            all_edge_attrs.append(item['input_edge_attr'][t])

        # Collect edges from all target timesteps
        for t in range(len(item['target_edge_index'])):
            all_edge_indices.append(item['target_edge_index'][t])
            all_edge_attrs.append(item['target_edge_attr'][t])

        # Concatenate and coalesce to remove duplicates
        if all_edge_indices:
            combined_edge_index = torch.cat(all_edge_indices, dim=1)
            combined_edge_attr = torch.cat(all_edge_attrs, dim=0)
            # Coalesce to merge duplicate edges (using mean for edge attributes)
            global_edge_index, global_edge_attr = coalesce(
                combined_edge_index,
                combined_edge_attr,
                num_nodes=item['num_nodes'],
                reduce='mean'
            )
        else:
            global_edge_index = torch.zeros((2, 0), dtype=torch.long)
            global_edge_attr = torch.zeros(0, dtype=torch.float32)

        global_graph_list.append(Data(
            edge_index=global_edge_index,
            edge_attr=global_edge_attr,
            num_nodes=item['num_nodes']
        ))

    global_graph = Batch.from_data_list(global_graph_list)

    return {
        'features': features,
        'targets': targets,
        'feature_indices': feature_indices.transpose(-2, -1),
        'target_indices': target_indices.transpose(-2, -1),
        'input_graphs': input_graphs,
        'target_graphs': target_graphs,
        'global_graph': global_graph,
        'composer_id': composer_ids,
        'composer_name': composer_names,
        'piano_roll': torch.stack([item['piano_roll'] for item in batch]),
        'voice_roll': torch.stack([item['voice_roll'] for item in batch]),
        'total_labels': torch.tensor([item['total_labels'] for item in batch]),
        'file_paths': file_paths,
        'node_mask': node_mask,
    }
