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


def save_generation(result, output_path):
    """Save generated output to file."""
    torch.save(result, output_path)
    print(f"\nSaved generated output to {output_path}")
    print(f"  - output shape: {result['output'].shape}")
    print(f"  - latent shape: {result['latent'].shape}")


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
    save_generation(result, args.output)

    print("\nGeneration complete!")
    print(f"\nNext steps:")
    print(f"1. Convert to MIDI (you'll need to implement this based on your data format)")
    print(f"2. Experiment with different temperatures and active_nodes_ratios")
    print(f"3. Generate multiple samples and select the best ones")


if __name__ == '__main__':
    main()
