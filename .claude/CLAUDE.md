# Aeolia: Autoregressive Music Generation with Spatio-Temporal Graph Neural Networks

## Communication Style Preferences

**Evidence-based technical assessment:**
- **Correctness over agreement/disagreement**: Technical accuracy matters most. Don't reflexively agree or disagree - verify against actual code first
- **Ground claims in code reality**: When discussing what the system does, read the actual implementation. Comments and docs can be stale/wrong
- **Show your work**: If disagreeing, cite specific lines of code, measurements, or concrete technical reasons. No bare assertions
- **Update on evidence**: If claims contradict your understanding, check the codebase before assuming they're wrong. Your mental model may be outdated
- **Distinguish fact from preference**: Be clear whether you're discussing technical correctness (provable) vs. design tradeoffs (debatable)
- **No unnecessary praise**: Focus on problems and improvements, not compliments
- **Direct language**: "That won't work because [evidence]" instead of "That's interesting, but consider..."
- **Highlight risks**: Point out performance issues, maintainability problems, security concerns upfront with specific evidence
- **Prefer removing code**: If there's a way to solve a problem by removing code, prefer that. Don't create helper scripts unless explicitly asked 

## Project Overview

Aeolia is a deep learning system for classical music composition that adapts **ASTGCN (Attention-based Spatial-Temporal Graph Convolutional Networks)** from traffic forecasting to polyphonic music generation. The system learns from 14,782 classical compositions across 47 composers to generate new music autoregressively.

**Key Innovation**: Treats polyphonic music as dynamic graphs where:
- **Nodes** represent musical entities (pitches, voices, rhythms, composer)
- **Edges** encode harmonic relationships (voice connections, rhythmic patterns)
- **Time** captures melodic progressions through temporal attention

## Core Technologies

- **PyTorch 2.7.0+** with PyTorch Geometric for graph neural networks
- **torch_geometric_temporal 0.56.2+** for temporal graph operations
- **pretty_midi** and **mido** for MIDI processing
- **TensorBoard** for training visualization
- Device support: MPS (Apple Silicon), CUDA, CPU

## Architecture: PolyphonyGCN

### High-Level Flow
```
MIDI → Graph Representation → Encoder (3 ASTGCN blocks) → Autoregressive Bottleneck
→ Decoder (3 transposed conv blocks) → Log-softmax output
```

### Key Components

#### 1. **ChebConvAttention** ([astgcn.py:54-240](src/models/astgcn.py#L54-L240))
- Chebyshev spectral graph convolution with spatial attention
- K-order polynomial approximation (typically K=1-3)
- Learns which harmonic relationships matter

#### 2. **SpatialAttention** ([astgcn.py:245-378](src/models/astgcn.py#L245-L378))
- Multi-head attention over graph nodes (harmonic structure)
- Output: [batch, num_nodes, num_nodes] attention matrix
- Learns which notes are musically relevant at each timestep

#### 3. **TemporalAttention** ([astgcn.py:380-532](src/models/astgcn.py#L380-L532))
- Multi-head attention over time dimension (melodic patterns)
- **Causal masking** prevents looking into future timesteps
- Output: [batch, timesteps, timesteps] attention matrix

#### 4. **STAttention Block** ([astgcn.py:533-701](src/models/astgcn.py#L533-L701))
- Fuses spatial and temporal attention
- Flow: TemporalAttention → spatial weighting → ChebConvAttention → GELU
- Handles variable-size graphs with per-sample masking

#### 5. **ASTGCNBlock** ([astgcn.py:886-1126](src/models/astgcn.py#L886-L1126))
- Combines STAttention with temporal convolution
- **Encoder blocks**: Conv2d with striding for downsampling
- **Decoder blocks**: ConvTranspose2d for symmetric upsampling
- Residual connections with LayerNorm

#### 6. **AutoregressiveBottleneck** ([astgcn.py:705-883](src/models/astgcn.py#L705-L883))
- Operates at encoder's bottleneck (latent space)
- Training: Autoregressively regenerates encoder output with teacher forcing
- Inference: Enables unconditional generation from latent space

### Hierarchical Compression
```
Encoder:  234 nodes → 57 → 20 → 10 (bottleneck)
Decoder:  10 nodes → 20 → 57 → 128 (output)
```

## Data Representation

### Graph Structure (234 nodes total)
- **128 pitch nodes**: MIDI pitches 0-127
- **16 voice nodes**: Polyphonic part assignments
- **39 rhythm nodes**: Beat subdivisions (0.125s to ~5s)
- **1 composer node**: Style embedding

### Edge Types
1. **Pitch ↔ Voice**: Harmonic relationships
2. **Voice ↔ Rhythm**: Temporal articulation
3. **Pitch ↔ Composer**: Style influence
4. **Pitch ↔ Pitch**: All active pitches (complete subgraph)
5. **Self-loop**: Composer node (ensures non-empty graph)

### Data Format
- Preprocessed as `.npz` files with sparse event sequences
- Events: (pitch, start_time, end_time, velocity, voice, rhythm)
- Time resolution: 0.125 seconds per timestep
- Sequence length: 300 timesteps (configurable)
- Vocabulary: 286 tokens (pitches + voices + rhythms + composers + padding)

## Directory Structure

```
├── src/
│   ├── data/
│   │   ├── dataset.py              # MidiGraphDataset - loads preprocessed NPZ
│   │   ├── preprocessing.py        # Data cleaning and format conversion
│   │   └── loader.py               # DataLoader with custom collate
│   ├── models/
│   │   └── astgcn.py               # Complete PolyphonyGCN implementation
│   ├── training/
│   │   ├── trainer.py              # Main training loop with checkpointing
│   │   ├── losses.py               # FocalPoissonNLLLoss, FocalCrossEntropyLoss
│   │   └── metrics.py              # MusicGenerationMetrics, NoteOnsetMetrics
│   └── utils/
│       ├── graph_helpers.py        # Graph pooling and adjacency operations
│       ├── midi_tensor_converter.py # Bidirectional MIDI↔tensor conversion
│       ├── visualization.py        # Piano roll visualization, activation maps
│       └── tensorboard_logger.py   # Training monitoring
├── configs/
│   ├── default_config.py           # Configuration class with auto device detection
│   └── config.yml                  # YAML parameter overrides
├── scripts/
│   ├── train.py                    # Main training entry point
│   ├── evaluate.py                 # Model evaluation
│   ├── generate.py                 # Autoregressive music generation
│   ├── generate_from_latent.py     # Generation from learned representations
│   ├── export_midi.py              # Convert generated tensors to MIDI
│   ├── process_midi.py             # Batch MIDI preprocessing
│   └── balance_data.py             # Dataset balancing
├── data/
│   ├── raw/                        # Full dataset (14,782 compositions)
│   ├── raw_test/                   # Test subset (organized by composer)
│   └── raw_test_unbalanced/        # Alternative test split
├── tests/                          # Unit test suite
├── notebooks/
│   └── examples.ipynb              # Interactive usage examples
├── runs/                           # TensorBoard event logs
└── visualizations/                 # Generated visualization outputs
```

## Training Pipeline

### Hyperparameters (config.yml)
- Input dimension: 16 (feature embeddings)
- Hidden dimension: 256-512
- K (Chebyshev filter size): 1-3
- Periods (sequence length): 300 timesteps
- Batch size: 4-64 (device-dependent)
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 100
- Scheduler: Linear warmup + Cosine annealing

### Loss Functions
- **FocalPoissonNLLLoss**: Primary loss for note count prediction with focal weighting
- **FocalCrossEntropyLoss**: Auxiliary composer classification
- Node masking: Loss computed only on active nodes (excludes padding)

### Metrics
- Precision, Recall, F1-score
- Mean Squared Error (MSE)
- Note onset metrics (predicted vs. ground truth onset times)

## Teacher-Forced Autoregressive Training

- **Lags mechanism**: 8-step offset between input and target
  - Input: timesteps 0-19
  - Target: timesteps 8-27
- Training: Model sees ground truth previous timesteps
- Inference: Sequential autoregressive generation with sampling
- Bottleneck AR: Additional autoregressive constraint at latent level

## Usage Examples

```bash
# Training
python scripts/train.py --config configs/config.yml --data_dir data/raw_test

# Generation
python scripts/generate.py --checkpoint models/best_model.pth \
    --num_steps 200 --temperature 0.8 --top_p 0.9

# Latent generation (unconditional)
python scripts/generate_from_latent.py --checkpoint models/best_model.pth \
    --composer_idx 5 --num_steps 200

# Export to MIDI
python scripts/export_midi.py --input generated_music.pt --output output.mid
```

## Recent Development

Recent commits show active refinement:
- **14b2c2a**: Refactored decoder to use transposed convolutions for symmetric upsampling
- **f747729**: Added MIDI-tensor conversion utilities and TensorBoard logging
- **fbc9b91**: Added autoregressive bottleneck implementation with generation capabilities
- **dc56f44**: Refactored visualization to separate concerns from metrics
- **46a1668**: Fixed metrics calculation for log probability outputs

## Unique Design Features

1. **Spatio-Temporal Modeling**: Jointly learns harmonic (spatial) and melodic (temporal) patterns
2. **Graph Neural Networks**: Natural representation for polyphonic voice relationships
3. **Attention Mechanisms**: Interpretable learning of which notes matter at each timestep
4. **Variable-Length Support**: Handles compositions of different lengths and complexity
5. **Flexible Sampling**: Temperature, top-k, nucleus sampling for controlled generation
6. **Composer Conditioning**: Multi-composer training with style transfer capability
7. **Unconditional Generation**: Pure latent sampling via autoregressive bottleneck
8. **Symmetric Encoder-Decoder**: Transposed convolutions mirror encoder compression

## Common Gotchas

- **Node masking**: Always apply masks when computing losses/metrics to exclude inactive nodes
- **Graph construction**: Per-timestep graphs vs. global graph (currently using global)
- **Composer node**: Index 233, must be handled specially in outputs
- **Time resolution**: 0.125s per step, affects rhythm interpretation
- **Device selection**: Auto-detects MPS/CUDA, but can be overridden in config
- **Sequence padding**: Variable-length sequences require careful batching (see `temporal_graph_collate`)

## Key Hyperparameter Tuning

When experimenting, focus on:
- **num_nodes_in/out**: Controls hierarchical compression levels
- **K**: Chebyshev filter order (higher = larger receptive field)
- **hidden_dim**: Model capacity vs. memory tradeoff
- **periods_in/out**: Temporal downsampling ratios
- **temperature**: Controls diversity in generation (0.8-1.2)
- **top_p**: Nucleus sampling threshold (0.9 recommended)

## Development Workflow

1. **Data preprocessing**: Use `scripts/process_midi.py` to convert MIDI to .npz
2. **Training**: Run `scripts/train.py` with TensorBoard monitoring
3. **Evaluation**: Use `scripts/evaluate.py` for test set metrics
4. **Generation**: Try `scripts/generate.py` for autoregressive sampling
5. **Visualization**: Check `visualizations/` for piano rolls and attention maps
6. **Export**: Convert tensors to MIDI with `scripts/export_midi.py`

## Testing

Run unit tests with:
```bash
pytest tests/
```

Tests cover:
- Graph construction and batching
- Model forward pass shapes
- Loss computation with masking
- MIDI-tensor conversion
- Metrics calculation
