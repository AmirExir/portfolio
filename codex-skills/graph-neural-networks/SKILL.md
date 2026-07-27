---
name: graph-neural-networks
description: Build graph neural networks for topology-aware power-system prediction, classification, ranking, and state or violation estimation.
---

# Graph Neural Networks for Power Systems

## Use this skill when

Use this skill for node, edge, or graph prediction using grid topology, buses, branches, transformers, generators, loads, measurements, contingencies, or operating states.

## Required workflow

1. Define graph entities and prediction target.
2. Define node, edge, and global features with units.
3. Define how topology changes and parallel circuits are represented.
4. Establish non-graph baselines.
5. Prevent leakage across time, cases, contingencies, and near-duplicate topologies.
6. Train and evaluate across multiple seeds.
7. Test robustness to topology and operating-point changes.
8. Explain model outputs with engineering context.

## Graph construction rules

- Use stable IDs for buses and equipment.
- Preserve directionality where physically meaningful.
- Represent transformer ratios, phase shifts, ratings, status, and impedance explicitly when relevant.
- Document self-loops, normalization, and aggregation choices.
- Do not collapse parallel circuits without a justified aggregation rule.
- Handle islands and disconnected components intentionally.

## Baselines

Compare against:

- linear or tree-based tabular models
- MLP using the same features
- physics-derived screening metrics
- topology-agnostic heuristics

## Evaluation

Use task-appropriate metrics and include:

- performance by voltage level
- performance by contingency severity
- rare-event performance
- topology generalization
- calibration when probabilities are used

## Reproducibility

Store graph-building code, feature schema, normalization statistics, split definitions, model configuration, and checkpoints.

## Completion format

Report:

- Graph definition
- Features and labels
- Leakage controls
- Baselines
- Results
- Generalization limits
