#!/usr/bin/env python3
"""
Export generated sequences to MIDI files.

This script converts the tensor representations from the generative model
back into MIDI format for playback and analysis.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))


def tensor_to_piano_roll(features, feature_indices, config):
    """
    Convert generated tensors to piano roll format.

    Args:
        features: (batch, nodes, time) tensor of velocities
        feature_indices: (batch, nodes, time) tensor of indices
        config: Configuration object

    Returns:
        piano_roll: (time, 128) numpy array representing MIDI piano roll
    """
    batch_size, num_nodes, num_time = features.shape

    # For now, take first item from batch
    features_single = features[0].cpu().numpy()  # (nodes, time)
    indices_single = feature_indices[0].cpu().numpy()  # (nodes, time)

    # Initialize piano roll (time, 128 pitches)
    piano_roll = np.zeros((num_time, 128))

    # Map nodes to pitches
    # This depends on your encoding scheme
    # Here's a simplified version assuming nodes represent pitches
    for t in range(num_time):
        for n in range(num_nodes):
            velocity = features_single[n, t]
            index = int(indices_single[n, t])

            # If this is a pitch node (not voice or rhythm)
            if 0 <= index < 128 and velocity > 0:
                piano_roll[t, index] = velocity

    return piano_roll


def piano_roll_to_midi(piano_roll, output_path, time_step=0.125, velocity_threshold=0.1):
    """
    Convert piano roll to MIDI file.

    Args:
        piano_roll: (time, 128) array
        output_path: Path to save MIDI file
        time_step: Duration of each timestep in seconds
        velocity_threshold: Minimum velocity to consider as note-on
    """
    try:
        import pretty_midi
    except ImportError:
        print("Error: pretty_midi not installed. Install with: pip install pretty_midi")
        return

    # Create MIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an instrument (piano)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Track active notes
    active_notes = {}  # pitch -> (start_time, velocity)

    num_time, num_pitches = piano_roll.shape

    for t in range(num_time):
        current_time = t * time_step

        for pitch in range(num_pitches):
            velocity = piano_roll[t, pitch]

            # Note onset
            if velocity > velocity_threshold and pitch not in active_notes:
                active_notes[pitch] = (current_time, int(velocity * 127))

            # Note offset (velocity drops or end of sequence)
            elif pitch in active_notes and (velocity <= velocity_threshold or t == num_time - 1):
                start_time, note_velocity = active_notes[pitch]
                end_time = current_time

                # Create note
                note = pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)

                del active_notes[pitch]

    # Add instrument to MIDI
    midi.instruments.append(piano)

    # Write to file
    midi.write(str(output_path))
    print(f"MIDI file saved to {output_path}")
    print(f"Duration: {midi.get_end_time():.2f} seconds")
    print(f"Number of notes: {len(piano.notes)}")


def main():
    parser = argparse.ArgumentParser(description='Export generated sequences to MIDI')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to generated sequence (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output MIDI file path (default: same as input with .mid extension)')
    parser.add_argument('--time_step', type=float, default=0.125,
                        help='Duration of each timestep in seconds')
    parser.add_argument('--velocity_threshold', type=float, default=0.1,
                        help='Minimum velocity to consider as note-on')

    args = parser.parse_args()

    # Load generated sequence
    print(f"Loading generated sequence from {args.input}...")
    generated = torch.load(args.input, map_location='cpu')

    features = generated['features']
    feature_indices = generated['feature_indices']

    print(f"Sequence shape: {features.shape}")
    print(f"Number of timesteps: {generated.get('num_steps_generated', 'unknown')}")

    # Determine output path
    if args.output is None:
        args.output = Path(args.input).with_suffix('.mid')

    # Convert to piano roll
    print("\nConverting to piano roll...")
    piano_roll = tensor_to_piano_roll(features, feature_indices, config=None)

    # Convert to MIDI
    print("Converting to MIDI...")
    piano_roll_to_midi(
        piano_roll,
        args.output,
        time_step=args.time_step,
        velocity_threshold=args.velocity_threshold
    )

    print("\nExport complete!")


if __name__ == '__main__':
    main()
