import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, npz_dir, sequence_length=32):
        """
        Dataset for preprocessed MIDI files
        
        Args:
            npz_dir: Directory containing preprocessed .npz files
            sequence_length: Length of sequences to return
        """
        self.npz_dir = npz_dir
        self.sequence_length = sequence_length
        self.file_list = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        
        # Optional: Create an index that maps from position to (file_idx, start_idx)
        # This enables more efficient random access across the entire dataset
        self._build_index()
        
    def _build_index(self):
        """Create an index mapping sample_idx -> (file_idx, start_pos)"""
        self.index = []
        
        for file_idx, npz_file in enumerate(self.file_list):
            data = np.load(os.path.join(self.npz_dir, npz_file))
            piano_roll = data['piano_roll']
            
            # For each possible starting position in this file
            for start_pos in range(0, piano_roll.shape[1] - self.sequence_length + 1):
                self.index.append((file_idx, start_pos))
                
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_idx, start_pos = self.index[idx]
        npz_path = os.path.join(self.npz_dir, self.file_list[file_idx])
        
        # Load the data
        data = np.load(npz_path)
        piano_roll = data['piano_roll']
        
        # Extract the sequence
        sequence = piano_roll[:, start_pos:start_pos + self.sequence_length]
        
        # Convert to tensor
        return torch.FloatTensor(sequence)