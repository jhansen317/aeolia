# Aeolia: Autoregressive Music Generation with Graph Neural Networks

Aeolia is a deep learning project that applies **Attention-based Spatial-Temporal Graph Convolutional Networks (ASTGCN)** to autoregressive music generation. The system represents polyphonic MIDI as temporal graphs and generates new compositions by predicting the next timestep autoregressively.

*Named after the aeolian harp, an instrument that sounds when wind passes through its strings.*

## Overview

This project adapts the ASTGCN architecture (originally designed for traffic flow forecasting) to music generation. Musical compositions are represented as temporal graphs where:
- **Nodes**: MIDI pitches (0-127), voice assignments, and rhythm values
- **Edges**: Temporal connections between voices in polyphonic music
- **Node Features**: Pitch, voice, and rhythm embeddings
- **Task**: Autoregressive next-timestep prediction for music generation
- **Training**: Teacher forcing on 47 classical composers (14,782 compositions)

## Key Features

- **Graph-based Music Representation**: Converts polyphonic MIDI into dynamic graph structures
- **Autoregressive Generation**: Next-timestep prediction with teacher forcing during training
- **Flexible Sampling**: Temperature, top-k, and nucleus (top-p) sampling for controlled generation
- **Attention Mechanisms**: Spatial and temporal attention for learning musical patterns
- **Spatio-Temporal Modeling**: Captures both harmonic relationships (spatial) and melodic progressions (temporal)
- **Scalable Architecture**: Handles variable-length compositions and complex polyphonic textures

## Dataset

- **Composers**: 47 classical composers
- **Samples**: 14,782 preprocessed compositions
- **Format**: NumPy compressed arrays (.npz) containing graph representations
- **Time Resolution**: 0.125 seconds per time step
- **Graph Structure**: Variable number of nodes (active pitches) and edges (voice connections)

## Project Structure

```
aeolia/
├── src/
│   ├── data/
│   │   ├── dataset.py          # MidiGraphDataset for loading graph data
│   │   ├── preprocessing.py    # Data cleaning utilities
│   │   └── loader.py           # DataLoader for batching
│   ├── models/
│   │   └── astgcn.py           # ASTGCN implementation (PolyphonyGCN)
│   ├── training/
│   │   ├── trainer.py          # Training loop management
│   │   └── losses.py           # Custom loss functions
│   └── utils/
│       ├── graph_helpers.py    # Graph pooling operations
│       ├── metrics.py          # Performance metrics
│       └── visualization.py    # Piano roll & graph visualization
├── configs/
│   ├── default_config.py       # Configuration with auto device detection
│   └── config.yml              # YAML parameter overrides
├── scripts/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Model evaluation
│   ├── process_midi.py         # MIDI preprocessing
│   └── balance_data.py         # Dataset balancing
├── tests/                      # Unit tests
├── notebooks/                  # Example notebooks
├── CITATIONS.md                # Academic references
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.7.0+
- PyTorch Geometric 2.6.1+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jhansen317/aeolia.git
cd aeolia
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python scripts/train.py
```

Configuration options can be modified in [configs/default_config.py](configs/default_config.py) or [configs/config.yml](configs/config.yml).

Key hyperparameters:
- `input_dim`: Feature dimension (default: 16-64)
- `hidden_dim`: Hidden layer size (default: 256-512)
- `K`: Chebyshev filter size (default: 1-3)
- `num_blocks`: Number of ASTGCN blocks (default: 2-3)
- `periods`: Sequence length (default: 200-300)
- `batch_size`: Training batch size (default: 4-16)
- `learning_rate`: Optimizer learning rate (default: 0.001)
- `num_epochs`: Training iterations (default: 100)

### Evaluating a Model

```bash
python scripts/evaluate.py
```

### Generating Music

Generate new music autoregressively from a trained model:

```bash
# Generate from a trained checkpoint
python scripts/generate.py \
    --checkpoint models/best_model.pth \
    --num_steps 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --output generated_music.pt

# Export to MIDI
python scripts/export_midi.py \
    --input generated_music.pt \
    --output output.mid
```

**Generation parameters:**
- `--temperature`: Controls randomness (0.5-2.0, default 1.0)
  - Lower = more conservative, follows training data closely
  - Higher = more creative/random
- `--top_k`: Sample from top K most likely tokens (optional)
- `--top_p`: Nucleus sampling - sample from top tokens with cumulative probability > p (default 0.9)
- `--num_steps`: Number of timesteps to generate

### Data Preprocessing

```bash
python scripts/process_midi.py --input_dir /path/to/midi --output_dir data/raw
```

## Architecture

The system uses the **PolyphonyGCN** model from [src/models/astgcn.py](src/models/astgcn.py), which consists of:

1. **ChebConvAttention**: Chebyshev spectral graph convolution with attention
2. **SpatialAttention**: Multi-head attention over graph structure (harmonic relationships)
3. **TemporalAttention**: Multi-head attention over time dimension (melodic patterns)
4. **STAttentionBlock**: Spatio-temporal fusion block
5. **ASTGCNBlock**: Stacked attention blocks for hierarchical feature learning
6. **Output Layer**: Log-softmax over token vocabulary for next-timestep prediction
7. **Generation Interface**: Autoregressive sampling with temperature/top-k/nucleus controls

## Harmonization Learning via Voice Dropout

Aeolia incorporates **voice-level dropout** during training to explicitly learn harmonization. During each training step, individual voice nodes are randomly masked (zeroed out) across all timesteps with probability `voice_dropout_rate` (default 20%).

### Why Voice Dropout?

When a voice is masked:
- The model receives **no information** about that voice's notes
- Graph edges from the voice node remain, but carry no features
- The model must predict the voice's notes using only:
  - Other active voices (harmonic context)
  - Pitch and rhythm structure
  - Composer style embeddings
  - Temporal patterns learned through attention

This creates an implicit **harmonization objective**: the model learns to "fill in" missing voices based on musical context, similar to how a composer would harmonize a melody.

### Musical Benefits

1. **Improved polyphonic coherence**: Model learns voice interdependencies explicitly
2. **Better voice leading**: Forced to predict voice motion from harmonic context
3. **Robust generation**: Model less prone to generating harmonically incoherent voices
4. **Implicit counterpoint**: Learns contrapuntal relationships between voices

### Configuration

```yaml
# In config.yml
voice_dropout_rate: 0.2  # 20% chance each voice is masked (0.0 to disable)
```

Set to `0.0` to disable voice dropout and train with all voices visible.


## Citation

This project is based on the ASTGCN architecture. If you use this code, please cite the original paper:

```bibtex
@article{guo2019attention,
  title={Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={922--929},
  year={2019},
  doi={10.1609/aaai.v33i01.3301922}
}
```

See [CITATIONS.md](CITATIONS.md) for additional references and adaptation details.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

The codebase follows standard Python conventions:
- **Classes**: PascalCase (e.g., `PolyphonyGCN`, `MidiGraphDataset`)
- **Functions**: snake_case (e.g., `get_output_size`, `batch_to_device`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `GLOBAL_DROPOUT`, `DROPOUT_2D`)

### Device Support

The configuration automatically detects and uses the best available device:
- **Apple Silicon**: MPS (Metal Performance Shaders)
- **NVIDIA GPU**: CUDA
- **CPU**: Fallback for compatibility

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Original ASTGCN implementation by Guo et al.
- PyTorch Geometric team for graph neural network primitives
- Classical music MIDI datasets used for training
