# LSTM-Tree

```
├── acd
│   ├── acd: acd core algorithm
│   ├── processing: generate trees
│   ├── reproduce_figs: reproduce figures in the acd paper
│   └── visualization: tools for visualization
├── data
│   ├── acd_trees_128d: generated 128d sst trees
│   ├── acd_trees_512d: generated 512d sst trees
│   ├── acd_trees_512d_rand: generated 512d sst trees with random model
│   ├── gold_trees: gold sst trees
│   └── process.py: normalize labels to evaluate
├── eval
│   ├── label: evaluate with annotated labels
│   ├── nolabel: evaluate without annotated labels
│   └── recursive: evaluate with recursive neural networks
├── formula: calculate formula of acd algorithms
├── model
│   ├── snli: snli training scripts / models (arthur4)
│   └── sst: sst training scripts / models (arthur4)
```
