//! Domain gap: measure the distance between source and target domains.

use crate::{SourceTask, TargetTask};
use std::collections::HashMap;

/// Measures how different two domains are.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainGap {
    /// Overall gap (0.0 = identical, 1.0 = completely different).
    pub gap: f64,
    /// Feature overlap ratio.
    pub feature_overlap: f64,
    /// Importance divergence.
    pub importance_divergence: f64,
    /// Ternary bias disagreement.
    pub bias_disagreement: f64,
}

impl DomainGap {
    /// Compute domain gap between source and target.
    ///
    /// Runs in O(n + m) where n = source features and m = target features:
    /// the target features are indexed once in a hash map, then the source
    /// features are scanned once with O(1) lookups.
    pub fn compute(source: &SourceTask, target: &TargetTask) -> Self {
        // Index target features by name for O(1) lookup. If the target has
        // duplicate names, the last occurrence wins (same observable matching
        // behavior as the previous linear `find`).
        let target_index: HashMap<&str, &crate::task::FeatureDescriptor> = target
            .features
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect();

        let total_features = source.features.len() + target.features.len();

        let mut shared_count = 0usize;
        let mut importance_diff = 0.0f64;
        let mut bias_diff = 0usize;

        for sf in &source.features {
            if let Some(tf) = target_index.get(sf.name.as_str()) {
                shared_count += 1;
                importance_diff += (sf.importance - tf.importance).abs();
                if sf.ternary_bias != tf.ternary_bias {
                    bias_diff += 1;
                }
            }
        }

        // overlap = 2|shared| / (|S| + |T|), counting each source feature whose
        // name appears in the target (matches the previously documented ratio).
        let overlap = if total_features > 0 {
            2.0 * shared_count as f64 / total_features as f64
        } else {
            0.0
        };

        let importance_div = if shared_count > 0 {
            importance_diff / shared_count as f64
        } else {
            1.0
        };

        let bias_disagree = if shared_count > 0 {
            bias_diff as f64 / shared_count as f64
        } else {
            1.0
        };

        let gap = (1.0 - overlap) * 0.4 + importance_div * 0.3 + bias_disagree * 0.3;

        Self {
            gap: gap.clamp(0.0, 1.0),
            feature_overlap: overlap,
            importance_divergence: importance_div,
            bias_disagreement: bias_disagree,
        }
    }

    /// Returns true if the domains are closely related (gap < 0.3).
    pub fn is_close(&self) -> bool {
        self.gap < 0.3
    }

    /// Returns true if the domains are very different (gap > 0.7).
    pub fn is_distant(&self) -> bool {
        self.gap > 0.7
    }

    /// Recommend a transfer strategy based on domain gap.
    pub fn recommend_strategy(&self) -> &'static str {
        if self.gap < 0.2 {
            "direct_copy"
        } else if self.gap < 0.4 {
            "weighted_blend"
        } else if self.gap < 0.6 {
            "selective_transfer"
        } else {
            "progressive"
        }
    }
}
