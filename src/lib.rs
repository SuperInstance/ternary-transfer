//! # ternary-transfer
//!
//! Transfer learning for ternary agents — take knowledge learned in one environment
//! and apply it to a new one. Provides strategies for knowledge transfer, domain gap
//! measurement, and negative transfer detection.

pub mod domain;
pub mod negative;
pub mod score;
pub mod strategy;
pub mod task;

pub use domain::DomainGap;
pub use negative::NegativeTransferDetector;
pub use score::TransferScore;
pub use strategy::TransferStrategy;
pub use task::{SourceTask, TargetTask};

/// A ternary value: Negative, Neutral, or Positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Negative = -1,
    Neutral = 0,
    Positive = 1,
}

impl Ternary {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Negative),
            0 => Some(Ternary::Neutral),
            1 => Some(Ternary::Positive),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn to_f64(self) -> f64 {
        self as i8 as f64
    }
}

/// A knowledge matrix representing learned weights for ternary state features.
/// Each feature has a weight in [-1.0, 1.0] and a ternary sign.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeMatrix {
    /// Feature weights, values clamped to [-1.0, 1.0].
    weights: Vec<f64>,
    /// Number of features.
    feature_count: usize,
}

impl KnowledgeMatrix {
    /// Create a new knowledge matrix with the given weights.
    /// Weights are clamped to [-1.0, 1.0].
    pub fn new(weights: Vec<f64>) -> Self {
        let feature_count = weights.len();
        let weights: Vec<f64> = weights.into_iter().map(|w| w.clamp(-1.0, 1.0)).collect();
        Self { weights, feature_count }
    }

    /// Create a zero knowledge matrix of the given size.
    pub fn zeros(count: usize) -> Self {
        Self {
            weights: vec![0.0; count],
            feature_count: count,
        }
    }

    /// Create a random-ish knowledge matrix (deterministic from seed).
    pub fn from_seed(count: usize, seed: u64) -> Self {
        let mut weights = Vec::with_capacity(count);
        let mut s = seed;
        for _ in 0..count {
            // Simple LCG for deterministic pseudo-random values
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let raw = ((s >> 33) as i64) as f64 / (1i64 << 31) as f64;
            weights.push(raw.clamp(-1.0, 1.0));
        }
        Self { weights, feature_count: count }
    }

    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the ternary sign of each weight.
    pub fn ternary_signs(&self) -> Vec<Ternary> {
        self.weights
            .iter()
            .map(|&w| if w < -0.33 {
                Ternary::Negative
            } else if w > 0.33 {
                Ternary::Positive
            } else {
                Ternary::Neutral
            })
            .collect()
    }

    /// Compute cosine similarity with another knowledge matrix.
    pub fn cosine_similarity(&self, other: &KnowledgeMatrix) -> f64 {
        if self.feature_count != other.feature_count || self.feature_count == 0 {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..self.feature_count {
            dot += self.weights[i] * other.weights[i];
            norm_a += self.weights[i] * self.weights[i];
            norm_b += other.weights[i] * other.weights[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }

    /// Element-wise weighted blend with another matrix.
    pub fn blend(&self, other: &KnowledgeMatrix, alpha: f64) -> KnowledgeMatrix {
        let alpha = alpha.clamp(0.0, 1.0);
        let weights: Vec<f64> = self.weights.iter().zip(other.weights.iter())
            .map(|(&a, &b)| (a * alpha + b * (1.0 - alpha)).clamp(-1.0, 1.0))
            .collect();
        KnowledgeMatrix::new(weights)
    }
}

/// Result of a transfer operation.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// The transferred knowledge matrix.
    pub knowledge: KnowledgeMatrix,
    /// The transfer score.
    pub score: TransferScore,
    /// Whether negative transfer was detected.
    pub negative_detected: bool,
    /// Domain gap between source and target.
    pub domain_gap: DomainGap,
}
