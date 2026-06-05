# ternary-transfer

Transfer learning for ternary agents — take knowledge learned in one environment and apply it to a new one.

## Overview

This crate implements transfer learning in the context of ternary agents whose knowledge is represented as ternary values (Negative / Neutral / Positive). It provides strategies to transfer learned knowledge from a **source task** to a **target task**, measure the domain gap between them, score the effectiveness of the transfer, and detect when transfer hurts performance (negative transfer).

## Core Concepts

### Ternary Values

Knowledge is expressed using three-valued logic:

| Value | Meaning |
|-------|---------|
| `Negative` (-1) | Feature hurts performance |
| `Neutral` (0) | Feature has no effect |
| `Positive` (+1) | Feature helps performance |

### SourceTask & TargetTask

- **SourceTask**: The environment where knowledge was originally learned. Contains a knowledge matrix (learned weights) and feature descriptors.
- **TargetTask**: The new environment to transfer knowledge to. May have a baseline knowledge state.

### TransferStrategy

Four strategies for transferring knowledge:

| Strategy | Description |
|----------|-------------|
| `DirectCopy` | Copy source knowledge directly to the target |
| `WeightedBlend` | Blend source and target knowledge with a weight parameter α |
| `SelectiveTransfer` | Transfer only selected feature indices |
| `Progressive` | Iteratively blend with decaying α for gradual transfer |

### TransferScore

Measures how much the transfer helped (or hurt):

- **estimated_performance**: Predicted performance after transfer (0.0–1.0)
- **improvement**: Delta over baseline (positive = helpful)
- **relative_improvement**: Ratio of improvement to baseline
- **confidence**: How reliable the estimate is (based on feature overlap)

### DomainGap

Measures the distance between source and target domains:

- **gap**: Overall distance (0.0 = identical, 1.0 = completely different)
- **feature_overlap**: Ratio of shared features
- **importance_divergence**: How much feature importance differs
- **bias_disagreement**: How much ternary biases disagree

Recommends a strategy based on gap severity.

### NegativeTransferDetector

Detects when transfer would hurt performance. Provides risk levels (Low, Medium, High, Critical) and actionable recommendations.

## Quick Start

```rust
use ternary_transfer::*;

// Define features for source and target tasks
let features = vec![
    task::FeatureDescriptor {
        name: "speed".into(),
        importance: 0.8,
        ternary_bias: Ternary::Positive,
    },
    task::FeatureDescriptor {
        name: "size".into(),
        importance: 0.5,
        ternary_bias: Ternary::Negative,
    },
];

// Create source task with learned knowledge
let source = SourceTask::new(
    "maze-navigation",
    features.clone(),
    KnowledgeMatrix::new(vec![0.9, -0.7]),
);

// Create target task
let target = TargetTask::new("obstacle-course", features);

// Choose a transfer strategy
let strategy = TransferStrategy::WeightedBlend { alpha: 0.7 };

// Execute transfer
let result = strategy.execute(&source, &target);

println!("Estimated performance: {:.2}", result.score.estimated_performance);
println!("Improvement: {:.2}", result.score.improvement);
println!("Negative transfer: {}", result.negative_detected);
println!("Domain gap: {:.2}", result.domain_gap.gap);
```

## Requirements

- Pure Rust, no `unsafe`, no external dependencies
- Edition 2021
- MIT licensed

## Running Tests

```bash
cargo test
```

21 tests covering all core functionality.

## License

MIT

## See Also
- **ternary-fitness** — related
- **ternary-ensemble** — related
- **ternary-federated** — related
- **ternary-ga** — related
- **ternary-curriculum** — related

