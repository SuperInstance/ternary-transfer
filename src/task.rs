//! Source and target tasks for transfer learning.

use crate::{KnowledgeMatrix, Ternary};

/// A feature descriptor for a task domain.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDescriptor {
    /// Name of the feature. Used to match features across source and target.
    pub name: String,
    /// How important this feature is to its task. Expected in [0.0, 1.0];
    /// the absolute value is what the scoring math actually consumes, but
    /// callers should keep it non-negative.
    pub importance: f64,
    /// The expected ternary direction of this feature's weight.
    pub ternary_bias: Ternary,
}

/// The environment where knowledge was originally learned.
#[derive(Debug, Clone)]
pub struct SourceTask {
    /// Name of the source task/domain.
    pub name: String,
    /// Feature descriptors for this task.
    pub features: Vec<FeatureDescriptor>,
    /// Knowledge learned in this task.
    pub knowledge: KnowledgeMatrix,
    /// Performance achieved in the source task (0.0 to 1.0).
    pub source_performance: f64,
}

impl SourceTask {
    pub fn new(name: &str, features: Vec<FeatureDescriptor>, knowledge: KnowledgeMatrix) -> Self {
        Self {
            name: name.to_string(),
            features,
            knowledge,
            source_performance: 1.0,
        }
    }

    /// Get feature names.
    pub fn feature_names(&self) -> Vec<&str> {
        self.features.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get the importance vector.
    pub fn importance_vector(&self) -> Vec<f64> {
        self.features.iter().map(|f| f.importance).collect()
    }

    /// How many features are shared with another task (by name).
    pub fn shared_features(&self, other: &TargetTask) -> Vec<String> {
        let other_names: Vec<&str> = other.feature_names();
        self.features
            .iter()
            .filter(|f| other_names.contains(&f.name.as_str()))
            .map(|f| f.name.clone())
            .collect()
    }
}

/// The new environment to transfer knowledge to.
#[derive(Debug, Clone)]
pub struct TargetTask {
    /// Name of the target task/domain.
    pub name: String,
    /// Feature descriptors for this task.
    pub features: Vec<FeatureDescriptor>,
    /// Optional baseline knowledge (e.g., random initialization).
    pub baseline_knowledge: Option<KnowledgeMatrix>,
    /// Baseline performance without transfer (0.0 to 1.0).
    pub baseline_performance: f64,
}

impl TargetTask {
    pub fn new(name: &str, features: Vec<FeatureDescriptor>) -> Self {
        let feature_count = features.len();
        Self {
            name: name.to_string(),
            features,
            baseline_knowledge: Some(KnowledgeMatrix::zeros(feature_count)),
            baseline_performance: 0.0,
        }
    }

    pub fn with_baseline(mut self, knowledge: KnowledgeMatrix, performance: f64) -> Self {
        self.baseline_knowledge = Some(knowledge);
        self.baseline_performance = performance;
        self
    }

    pub fn feature_names(&self) -> Vec<&str> {
        self.features.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn feature_count(&self) -> usize {
        self.features.len()
    }
}
