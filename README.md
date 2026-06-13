# ternary-transfer

Transfer learning library for ternary agents — take knowledge learned in one environment and apply it to a new one, using the **{-1, 0, +1}** trit algebra as the shared representation for feature weights and domain comparisons.

## Why It Matters

Transfer learning is the single biggest lever for reducing sample complexity in RL and agent-based systems. When two tasks share structure, a good transfer can cut learning time by orders of magnitude. When they don't, **negative transfer** — applying irrelevant or misleading knowledge — can set you back further than starting from scratch.

This crate provides a principled framework for that decision: measure the **domain gap** between source and target, score the expected benefit, and detect negative transfer before it corrupts your agent.

## How It Works

### Knowledge Representation

Each agent's learned knowledge is stored as a `KnowledgeMatrix` — a vector of weights in **[-1.0, 1.0]**, each with a ternary sign (Negative / Neutral / Positive):

```
w_i ∈ [-1, 1],  sign(w_i) ∈ {−1, 0, +1}
```

The ternary sign is thresholded at ±0.33, matching the three-valued logic used throughout the ternary ecosystem.

### Domain Gap

The domain gap between source **S** and target **T** is a weighted metric:

```
gap(S, T) = 0.4 · (1 − overlap) + 0.3 · Δ_importance + 0.3 · Δ_bias
```

Where:
- **overlap** = 2|shared features| / (|S| + |T|) — Jaccard-like feature overlap
- **Δ_importance** = mean absolute difference in feature importance over shared features
- **Δ_bias** = fraction of shared features where ternary bias disagrees

**Complexity:** O(n + m) where n = source features, m = target features.

### Transfer Strategies

| Strategy | Formula | When to use |
|---|---|---|
| **DirectCopy** | w_target[i] ← w_source[i] | gap < 0.2 |
| **WeightedBlend** | w ← α·w_source + (1−α)·w_target | gap < 0.4 |
| **SelectiveTransfer** | w_target[idx] ← w_source[idx] only | gap < 0.6 |
| **Progressive** | w ← blend, α *= decay per step | gap ≥ 0.6 |

### Transfer Score

Estimated performance after transfer:

```
P_est = clamp(P_baseline + 0.3 · alignment + 0.2 · P_source, 0, 1)
```

Where alignment rewards matching the target's ternary bias weighted by feature importance.

### Negative Transfer Detection

Negative transfer is flagged when:
- `improvement < −0.05`, OR
- `domain_gap > 0.7` AND `confidence < 0.3`

Risk levels: **Low** (gap < 0.4) → **Medium** (< 0.6) → **High** (< 0.8) → **Critical** (≥ 0.8).

## Quick Start

```rust
use ternary_transfer::*;

// Define source task with learned knowledge
let source = SourceTask::new(
    "navigation-v1",
    vec![
        FeatureDescriptor { name: "speed".into(), importance: 0.8, ternary_bias: Ternary::Positive },
        FeatureDescriptor { name: "obstacle".into(), importance: 0.9, ternary_bias: Ternary::Negative },
    ],
    KnowledgeMatrix::new(vec![0.9, -0.8]),
);

// Define target task
let target = TargetTask::new(
    "navigation-v2",
    vec![
        FeatureDescriptor { name: "speed".into(), importance: 0.7, ternary_bias: Ternary::Positive },
        FeatureDescriptor { name: "obstacle".into(), importance: 0.85, ternary_bias: Ternary::Negative },
    ],
);

// Execute transfer
let result = TransferStrategy::WeightedBlend { alpha: 0.7 }
    .execute(&source, &target);

println!("Improvement: {:.3}", result.score.improvement);
println!("Negative transfer: {}", result.negative_detected);
```

## API

| Type | Purpose |
|---|---|
| `SourceTask` | Domain where knowledge was originally learned |
| `TargetTask` | New domain to transfer knowledge into |
| `KnowledgeMatrix` | Weight vector with ternary sign classification |
| `TransferStrategy` | Enum: DirectCopy, WeightedBlend, SelectiveTransfer, Progressive |
| `DomainGap` | Measures feature overlap, importance divergence, bias disagreement |
| `TransferScore` | Estimated performance, improvement, confidence |
| `NegativeTransferDetector` | Detects and risk-assesses negative transfer |
| `TransferResult` | Bundled output: knowledge + score + gap + negative flag |

## Architecture Notes

The ternary ecosystem rests on a **conservation law**: the sum of agent population fractions γ (choose) + η (avoid) must equal the total active population, with the remainder in the neutral (unknown) state. Transfer learning shifts these fractions by moving weights — a DirectCopy pushes the target toward the source's γ+η distribution, while a WeightedBlend interpolates between two distributions. The domain gap quantifies how far apart those distributions are.

`KnowledgeMatrix::cosine_similarity` provides the vector-space view of this relationship:

```
cos(θ) = (w_S · w_T) / (‖w_S‖ · ‖w_T‖)
```

This is the same inner-product geometry used in the correlation metrics of `ternary-tuple` and the ballot tallies of `warp-ternary-vote`.

## References

- Pan, S. J. & Yang, Q. (2010). *"A Survey on Transfer Learning."* IEEE TKDD.
- Rosenstein, M. T. et al. (2005). *"Transfer Learning with Risk of Negative Transfer."* ICML.
- Yosinski, J. et al. (2014). *"How Transferable Are Features in Deep Neural Networks?"* NeurIPS.
- Torrey, L. & Shavlik, J. (2010). *"Transfer Learning."* Handbook of Research on Machine Learning.

## License

MIT
