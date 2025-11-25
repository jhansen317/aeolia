# Citations and References

This project builds upon and adapts several key works in the field of graph neural networks and spatio-temporal modeling.

## Primary Architecture Reference

The core architecture of this project is based on the **Attention-based Spatial-Temporal Graph Convolutional Network (ASTGCN)** model:

**Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019).** Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. *AAAI Conference on Artificial Intelligence*, 33(01), 922-929. https://doi.org/10.1609/aaai.v33i01.3301922

### BibTeX Entry
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

### Adaptation Notes

The original ASTGCN model was designed for traffic flow forecasting on spatial road networks. This project adapts the ASTGCN architecture for **music composition analysis** with the following key modifications:

1. **Domain Shift**: From traffic networks to musical graph structures
   - Nodes represent MIDI pitches (0-127) instead of traffic sensors
   - Edges represent temporal voice connections instead of road segments
   - Node features include pitch, voice, and rhythm embeddings

2. **Task Adaptation**: From regression (traffic prediction) to classification (composer identification)
   - Modified output layers for multi-class classification
   - Adapted loss functions for categorical targets

3. **Graph Structure**: From static spatial graphs to dynamic music graphs
   - Time-varying graph structures representing polyphonic music
   - Variable-length sequences based on composition duration

## Additional References

### Graph Neural Networks
- **Kipf, T. N., & Welling, M. (2017).** Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
- **Defferrard, M., Bresson, X., & Vandergheynst, P. (2016).** Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. *NeurIPS*.

### Attention Mechanisms
- **Vaswani, A., et al. (2017).** Attention Is All You Need. *NeurIPS*.

### Music Information Retrieval
- References to be added as the project develops specific MIR techniques.

---

## Acknowledgments

This implementation uses:
- **PyTorch** and **PyTorch Geometric** for deep learning on graphs
- **NumPy** for numerical computations
- **MIDI processing libraries** for music data preprocessing
