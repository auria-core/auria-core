use crate::error::{AuriaError, AuriaResult};
use crate::shard::{ExpertId, ShardId};
use crate::tensor::{Tensor, TensorLayout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub expert_ids: Vec<ExpertId>,
    pub confidence_scores: Vec<f32>,
    pub gating_weights: Vec<f32>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRoutingInfo {
    pub expert_id: ExpertId,
    pub shard_ids: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
    pub routing_score: f32,
    pub last_accessed: u64,
    pub hit_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfiguration {
    pub top_k: u32,
    pub temperature: f32,
    pub diversity_penalty: f32,
    pub use_caching: bool,
    pub cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub average_confidence: f32,
    pub routing_time_ms: u64,
}

pub struct Router {
    config: RoutingConfiguration,
    expert_registry: Arc<RwLock<HashMap<ExpertId, ExpertRoutingInfo>>>,
    cache: Arc<RwLock<HashMap<String, RoutingDecision>>>,
    stats: Arc<RwLock<RoutingStats>>,
}

impl Router {
    pub fn new(config: RoutingConfiguration) -> Self {
        Self {
            config,
            expert_registry: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RoutingStats {
                total_requests: 0,
                cache_hits: 0,
                average_confidence: 0.0,
                routing_time_ms: 0,
            })),
        }
    }

    pub fn route(
        &self,
        input_embedding: &[f32],
        model_state: &ModelState,
    ) -> AuriaResult<RoutingDecision> {
        let start_time = Instant::now();
        let start_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut stats = self.stats.write().unwrap();
        stats.total_requests += 1;

        // Check cache first if enabled
        if self.config.use_caching {
            let cache_key = self.generate_cache_key(input_embedding, model_state);
            if let Some(cached_decision) = self.cache.read().unwrap().get(&cache_key) {
                if start_timestamp - cached_decision.timestamp < self.config.cache_ttl_seconds {
                    stats.cache_hits += 1;
                    return Ok(cached_decision.clone());
                }
            }
        }

        // Calculate routing scores for all experts
        let mut expert_scores = Vec::new();
        let expert_registry = self.expert_registry.read().unwrap();

        for (expert_id, expert_info) in expert_registry.iter() {
            let score = self.calculate_routing_score(input_embedding, model_state, expert_info);
            expert_scores.push((expert_id.clone(), score, expert_info.routing_score));
        }

        // Sort by combined score (routing score + gating weights)
        expert_scores.sort_by(|a, b| {
            let combined_a = a.1 * a.2;
            let combined_b = b.1 * b.2;
            combined_b
                .partial_cmp(&combined_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top-K experts
        let top_k = self.config.top_k as usize;
        let selected_experts: Vec<_> = expert_scores.into_iter().take(top_k).collect();

        // Apply temperature and diversity
        let mut expert_ids = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut gating_weights = Vec::new();

        for (expert_id, score, gating_weight) in selected_experts {
            expert_ids.push(expert_id);
            confidence_scores.push(score);
            gating_weights.push(gating_weight);
        }

        // Calculate average confidence
        if !confidence_scores.is_empty() {
            stats.average_confidence =
                confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32;
        }

        // Calculate routing time
        stats.routing_time_ms = start_time.elapsed().as_millis() as u64;

        let decision = RoutingDecision {
            expert_ids,
            confidence_scores,
            gating_weights,
            timestamp: start_time.elapsed().as_secs(),
        };

        // Cache the decision if enabled
        if self.config.use_caching {
            let cache_key = self.generate_cache_key(input_embedding, model_state);
            self.cache
                .write()
                .unwrap()
                .insert(cache_key, decision.clone());
        }

        Ok(decision)
    }

    pub fn add_expert(&self, expert_info: ExpertRoutingInfo) {
        let mut expert_registry = self.expert_registry.write().unwrap();
        expert_registry.insert(expert_info.expert_id.clone(), expert_info);
    }

    pub fn get_stats(&self) -> RoutingStats {
        self.stats.read().unwrap().clone()
    }

    pub fn update_routing_configuration(&mut self, config: RoutingConfiguration) {
        self.config = config;
    }

    fn calculate_routing_score(
        &self,
        input_embedding: &[f32],
        _model_state: &ModelState,
        expert_info: &ExpertRoutingInfo,
    ) -> f32 {
        // Simple routing algorithm: dot product of input embedding with expert's routing vector
        let mut score = 0.0;
        for (i, &value) in input_embedding.iter().enumerate() {
            if i < expert_info.routing_score as usize {
                score += value * expert_info.routing_score;
            }
        }

        // Apply temperature
        score /= self.config.temperature;

        // Apply diversity penalty (using timestamp)
        let now_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let age_factor = ((now_timestamp - expert_info.last_accessed) as f32 / 3600.0).min(1.0);
        score *= 1.0 - (age_factor * self.config.diversity_penalty);

        // Clamp score between 0 and 1
        score.max(0.0).min(1.0)
    }

    fn generate_cache_key(&self, input_embedding: &[f32], model_state: &ModelState) -> String {
        // Generate a hash-based cache key
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &value in input_embedding {
            hasher.write_u32(value.to_bits());
        }
        hasher.write_u32(model_state.position);
        format!("{:x}", hasher.finish())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub position: u32,
    pub kv_cache: Vec<Tensor>,
    pub attention_mask: Vec<f32>,
    pub past_key_values: Option<PastKeyValues>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PastKeyValues {
    pub key: Vec<Tensor>,
    pub value: Vec<Tensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    pub input_tokens: Vec<String>,
    pub current_position: u32,
    pub sequence_length: u32,
    pub batch_size: u32,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRoutingPolicy {
    pub max_experts_per_step: u32,
    pub min_experts_per_step: u32,
    pub routing_algorithm: String,
    pub enable_caching: bool,
    pub cache_size: u32,
}

impl Router {
    pub fn create_routing_context(
        input_tokens: &[String],
        current_position: u32,
        batch_size: u32,
    ) -> RoutingContext {
        RoutingContext {
            input_tokens: input_tokens.to_vec(),
            current_position,
            sequence_length: input_tokens.len() as u32,
            batch_size,
            device: "cpu".to_string(), // Default device
        }
    }

    pub fn apply_routing_policy(&self, context: &RoutingContext) -> ExpertRoutingPolicy {
        // Simple policy based on context
        let max_experts = match context.batch_size {
            1 => 4,
            2..=8 => 8,
            _ => 16,
        };

        ExpertRoutingPolicy {
            max_experts_per_step: max_experts,
            min_experts_per_step: (max_experts / 2).max(1),
            routing_algorithm: "dot_product".to_string(),
            enable_caching: true,
            cache_size: 1000,
        }
    }

    pub fn validate_routing_decision(
        &self,
        decision: &RoutingDecision,
        policy: &ExpertRoutingPolicy,
    ) -> AuriaResult<()> {
        // Check if number of experts is within policy limits
        let num_experts = decision.expert_ids.len() as u32;
        let min_experts = policy.min_experts_per_step;
        if num_experts < min_experts {
            return Err(AuriaError::ExecutionError(format!(
                "Too few experts selected ({}), minimum required is {}",
                num_experts, min_experts
            )));
        }

        let max_experts = policy.max_experts_per_step;
        if num_experts > max_experts {
            return Err(AuriaError::ExecutionError(format!(
                "Too many experts selected ({}), maximum allowed is {}",
                num_experts, max_experts
            )));
        }

        // Check confidence scores
        for &score in &decision.confidence_scores {
            if score < 0.0 || score > 1.0 {
                return Err(AuriaError::ExecutionError(format!(
                    "Invalid confidence score: {}",
                    score
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_basic_routing() {
        let config = RoutingConfiguration {
            top_k: 4,
            temperature: 1.0,
            diversity_penalty: 0.1,
            use_caching: true,
            cache_ttl_seconds: 60,
        };

        let router = Router::new(config);

        // Add some test experts
        for i in 0..10 {
            let mut expert_id_bytes = [0u8; 32];
            expert_id_bytes[0] = i as u8;
            let expert_info = ExpertRoutingInfo {
                expert_id: ExpertId(expert_id_bytes),
                shard_ids: vec![],
                tensor_layout: TensorLayout {
                    offset: 0,
                    stride: 1,
                    shape: vec![128, 128],
                },
                routing_score: (i as f32 + 1.0) / 10.0,
                last_accessed: 0,
                hit_count: 0,
            };
            router.add_expert(expert_info);
        }

        // Create test input
        let input_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let model_state = ModelState {
            position: 0,
            kv_cache: vec![],
            attention_mask: vec![1.0; 5],
            past_key_values: None,
        };

        // Perform routing
        let decision = router.route(&input_embedding, &model_state).unwrap();

        // Verify results
        assert_eq!(decision.expert_ids.len(), 4);
        assert_eq!(decision.confidence_scores.len(), 4);
        assert_eq!(decision.gating_weights.len(), 4);

        // Check stats
        let stats = router.get_stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.cache_hits, 0);
        assert!(stats.average_confidence > 0.0);
        assert!(stats.routing_time_ms > 0);
    }

    #[test]
    fn test_routing_policy() {
        let config = RoutingConfiguration {
            top_k: 8,
            temperature: 0.7,
            diversity_penalty: 0.2,
            use_caching: false,
            cache_ttl_seconds: 30,
        };

        let router = Router::new(config);

        // Create routing context
        let context =
            Router::create_routing_context(&vec!["hello".to_string(), "world".to_string()], 2, 1);

        // Apply routing policy
        let policy = router.apply_routing_policy(&context);

        // Verify policy
        assert!(policy.max_experts_per_step >= policy.min_experts_per_step);
        assert!(policy.enable_caching);
        assert!(policy.cache_size > 0);
    }

    #[test]
    fn test_routing_decision_validation() {
        let config = RoutingConfiguration {
            top_k: 4,
            temperature: 1.0,
            diversity_penalty: 0.1,
            use_caching: true,
            cache_ttl_seconds: 60,
        };

        let router = Router::new(config);

        // Create valid routing decision
        let valid_decision = RoutingDecision {
            expert_ids: vec![ExpertId([1u8; 32]), ExpertId([2u8; 32])],
            confidence_scores: vec![0.8, 0.7],
            gating_weights: vec![0.9, 0.8],
            timestamp: 12345,
        };

        let policy = ExpertRoutingPolicy {
            max_experts_per_step: 4,
            min_experts_per_step: 1,
            routing_algorithm: "dot_product".to_string(),
            enable_caching: true,
            cache_size: 100,
        };

        // Validate decision
        let result = router.validate_routing_decision(&valid_decision, &policy);
        assert!(result.is_ok());

        // Create invalid decision (too few experts)
        let invalid_decision = RoutingDecision {
            expert_ids: vec![ExpertId([1u8; 32])],
            confidence_scores: vec![0.8],
            gating_weights: vec![0.9],
            timestamp: 12345,
        };

        let result = router.validate_routing_decision(&invalid_decision, &policy);
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("Too few experts selected"));
    }
}
