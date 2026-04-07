use crate::shard::{ExpertId, Hash, LicenseHash, ShardId};
use crate::tensor::{Tensor, TensorLayout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub expert_id: ExpertId,
    pub shards: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
    pub metadata: ExpertMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetadata {
    pub created_at: u64,
    pub version: u32,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub license_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDefinition {
    pub expert_id: ExpertId,
    pub shard_ids: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
    pub required_license: Option<LicenseHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertAssembly {
    pub expert_id: ExpertId,
    pub assembled_tensor: Tensor,
    pub assembly_timestamp: u64,
    pub deterministic_hash: Hash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertCacheEntry {
    pub expert_id: ExpertId,
    pub tensor: Tensor,
    pub last_used_timestamp: u64,
    pub hit_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertExecution {
    pub expert_id: ExpertId,
    pub input: Tensor,
    pub output: Tensor,
    pub execution_time_ms: u64,
    pub memory_usage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRegistry {
    pub experts: HashMap<ExpertId, ExpertDefinition>,
    pub assemblies: HashMap<ExpertId, ExpertAssembly>,
}

impl ExpertRegistry {
    pub fn new() -> Self {
        Self {
            experts: HashMap::new(),
            assemblies: HashMap::new(),
        }
    }

    pub fn add_expert(&mut self, expert: ExpertDefinition) {
        self.experts
            .insert(expert.expert_id.clone(), expert.clone());
    }

    pub fn get_expert(&self, expert_id: &ExpertId) -> Option<&ExpertDefinition> {
        self.experts.get(expert_id)
    }

    pub fn get_assembly(&self, expert_id: &ExpertId) -> Option<&ExpertAssembly> {
        self.assemblies.get(expert_id)
    }

    pub fn register_assembly(&mut self, assembly: ExpertAssembly) {
        self.assemblies.insert(assembly.expert_id.clone(), assembly);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRoutingDecision {
    pub expert_id: ExpertId,
    pub confidence_score: f32,
    pub gating_weights: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertExecutionPolicy {
    pub max_concurrent_executions: u32,
    pub memory_limit_bytes: u64,
    pub timeout_ms: u64,
    pub priority: u8,
}

pub type ExpertResult<T> = std::result::Result<T, ExpertError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertError {
    InvalidShard,
    AssemblyFailed,
    ExecutionFailed,
    MemoryLimitExceeded,
    Timeout,
    InvalidInput,
    InvalidOutput,
    NotAssembled,
    InvalidLayout,
    StorageError(String),
}
