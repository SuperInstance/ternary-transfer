//! Negative transfer detector: detect when transfer hurts performance.

use crate::{DomainGap, TransferScore};

/// Detects negative transfer — when transferred knowledge hurts target performance.
#[derive(Debug, Clone)]
pub struct NegativeTransferDetector {
    /// Threshold for declaring negative transfer (improvement below this is negative).
    pub threshold: f64,
    /// Minimum domain gap to consider as a risk factor.
    pub gap_risk_threshold: f64,
}

impl NegativeTransferDetector {
    pub fn new() -> Self {
        Self {
            threshold: -0.05,
            gap_risk_threshold: 0.6,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_gap_threshold(mut self, threshold: f64) -> Self {
        self.gap_risk_threshold = threshold;
        self
    }

    /// Detect negative transfer from score and domain gap.
    pub fn detect(score: &TransferScore, gap: &DomainGap) -> bool {
        // Negative if improvement is below threshold
        if score.improvement < -0.05 {
            return true;
        }
        // Also negative if high domain gap AND low confidence
        if gap.gap > 0.7 && score.confidence < 0.3 {
            return true;
        }
        false
    }

    /// Full analysis with this detector's thresholds.
    pub fn analyze(&self, score: &TransferScore, gap: &DomainGap) -> NegativeTransferReport {
        let detected = score.improvement < self.threshold
            || (gap.gap > self.gap_risk_threshold && score.confidence < 0.3);

        let risk_level = if gap.gap > 0.8 {
            RiskLevel::Critical
        } else if gap.gap > 0.6 {
            RiskLevel::High
        } else if gap.gap > 0.4 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        let recommendation = if detected {
            "Abort transfer — negative impact likely. Consider learning from scratch."
        } else if risk_level == RiskLevel::High || risk_level == RiskLevel::Critical {
            "High risk — use progressive or selective transfer with caution."
        } else if risk_level == RiskLevel::Medium {
            "Moderate risk — weighted blend recommended."
        } else {
            "Low risk — direct copy or weighted blend should work well."
        };

        NegativeTransferReport {
            negative_detected: detected,
            risk_level,
            recommendation: recommendation.to_string(),
            improvement: score.improvement,
            domain_gap: gap.gap,
        }
    }
}

impl Default for NegativeTransferDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk level for negative transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Report from negative transfer analysis.
#[derive(Debug, Clone)]
pub struct NegativeTransferReport {
    pub negative_detected: bool,
    pub risk_level: RiskLevel,
    pub recommendation: String,
    pub improvement: f64,
    pub domain_gap: f64,
}
