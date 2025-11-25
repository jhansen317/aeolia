from torch.utils.data import DataLoader
from .dataset import MIDIDataset

def get_midi_dataloader(npz_dir, batch_size=32, sequence_length=32, num_workers=4):
    """Create a DataLoader for MIDI data"""
    dataset = MIDIDataset(npz_dir, sequence_length)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )