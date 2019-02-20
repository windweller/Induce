# LSTM-Tree

```
├── acd
│   ├── acd: acd core algorithm
│   ├── processing: generate trees
│   ├── reproduce_figs: reproduce figures in the acd paper
│   └── visualization: tools for visualization
├── data
│   ├── acd_trees_128d: generated sst trees with trained 128d-model
│   ├── acd_trees_512d: generated sst trees with trained 512d-model
│   ├── acd_trees_512d_rand: generated sst trees with random 512d-model
│   ├── gold_trees: gold sst trees
│   └── process.py: normalize labels for evaluation
├── eval
│   ├── label: evaluate with annotated labels
│   ├── nolabel: evaluate without annotated labels
│   └── recursive: evaluate with recursive neural networks
├── formula: calculate formula of acd algorithms
├── model
    └── sst: sst training scripts
```
