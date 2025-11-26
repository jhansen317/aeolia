#!/usr/bin/env python3
"""
Generate music from pure latent space (unconditional generation).

This script demonstrates sampling from the learned latent space to generate
music without conditioning on any input sequence.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.models.astgcn import PolyphonyGCN
from configs.default_config import DefaultConfig
from src.utils.midi_tensor_converter import tensor_to_midi, batch_tensor_to_midi


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    model = PolyphonyGCN(config).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    return model


def generate_from_latent(
    model,
    batch_size=1,
    temperature=1.0,
    active_nodes_ratio=0.3,
):
    """
    Generate music from pure latent space.

    Args:
        model: Trained PolyphonyGCN model with autoregressive bottleneck
        batch_size: Number of samples to generate
        temperature: Sampling temperature (higher = more random)
        active_nodes_ratio: Ratio of nodes to keep active (0.0 to 1.0)

    Returns:
        result: Dictionary with generated output and latent representation
    """
    print(f"\nGenerating from latent space...")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"Active nodes ratio: {active_nodes_ratio}")

    with torch.no_grad():
        result = model.generate_from_latent(
            batch_size=batch_size,
            temperature=temperature,
            active_nodes_ratio=active_nodes_ratio,
            global_graph=None  # Will use default structure
        )

    return result


def save_generation(result, output_path, config, convert_to_midi=True, midi_output_dir=None):
    """Save generated output to file and optionally convert to MIDI."""
    torch.save(result, output_path)
    print(f"\nSaved generated output to {output_path}")
    print(f"  - output shape: {result['output'].shape}")
    print(f"  - latent shape: {result['latent'].shape}")

    if convert_to_midi:
        # Determine MIDI output directory
        if midi_output_dir is None:
            midi_output_dir = Path(output_path).parent / 'generated_midi'
        else:
            midi_output_dir = Path(midi_output_dir)

        midi_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting to MIDI...")
        print(f"  - Output directory: {midi_output_dir}")

        # Extract the output tensor (batch_size, num_nodes, seq_length)
        # Need to transpose to (batch_size, seq_length, num_nodes)
        output = result['output']
        if output.dim() == 3:
            # Assuming shape is (batch_size, num_nodes, seq_length)
            output = output.transpose(1, 2)

        batch_size = output.shape[0]

        # Convert batch to MIDI files
        midi_objects = batch_tensor_to_midi(
            features=output,
            output_dir=midi_output_dir,
            prefix="latent_gen",
            time_step=0.25,
            num_pitches=config.num_pitches,
            num_voices=config.num_voices,
            tempo=120.0,
            program=0,  # Acoustic Grand Piano
            velocity_threshold=0.01
        )

        print(f"  - Generated {len(midi_objects)} MIDI file(s)")
        for i in range(batch_size):
            print(f"    - {midi_output_dir / f'latent_gen_{i:04d}.mid'}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate music from latent space using trained ASTGCN model'
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default=str(project_root / 'configs' / 'config.yml'),
                        help='Path to config file')

    # Generation arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--active_nodes_ratio', type=float, default=0.3,
                        help='Ratio of nodes to keep active (0.0 to 1.0)')

    # Output arguments
    parser.add_argument('--output', type=str,
                        default=str(project_root / 'generated_latent.pt'),
                        help='Output path for generated sequence')
    parser.add_argument('--midi_output_dir', type=str,
                        default=None,
                        help='Output directory for MIDI files (default: generated_midi/ in same dir as output)')
    parser.add_argument('--no_midi', action='store_true',
                        help='Skip MIDI conversion')

    args = parser.parse_args()

    # Load configuration
    config = DefaultConfig(project_root=project_root, config_path=args.config)
    print(f"Using device: {config.device}")

    # Check if autoregressive bottleneck is enabled
    if not config.use_autoregressive_bottleneck:
        print("\nERROR: Autoregressive bottleneck is not enabled in config!")
        print("To use latent space generation, set use_autoregressive_bottleneck: true")
        print("in your config file and retrain the model.")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, config)

    # Generate from latent space
    result = generate_from_latent(
        model=model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        active_nodes_ratio=args.active_nodes_ratio,
    )

    # Save output
    save_generation(
        result=result,
        output_path=args.output,
        config=config,
        convert_to_midi=not args.no_midi,
        midi_output_dir=args.midi_output_dir
    )

    print("\nGeneration complete!")
    if not args.no_midi:
        print(f"\nNext steps:")
        print(f"1. Listen to the generated MIDI files")
        print(f"2. Experiment with different temperatures and active_nodes_ratios")
        print(f"3. Generate multiple samples and select the best ones")
    else:
        print(f"\nNext steps:")
        print(f"1. Convert to MIDI using the tensor_to_midi utilities")
        print(f"2. Experiment with different temperatures and active_nodes_ratios")
        print(f"3. Generate multiple samples and select the best ones")


if __name__ == '__main__':
    main()
