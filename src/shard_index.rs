// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Shard Index for AURIA Runtime Core.
//     Maintains mapping between shards and experts, providing fast lookup
//     of expert definitions and shard relationships.

use super::{AuriaError, AuriaResult, Expert, ExpertId, Shard, ShardId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDefinition {
    pub expert_id: ExpertId,
    pub shard_ids: Vec<ShardId>,
    pub tensor_layout: super::TensorLayout,
}

pub struct ShardIndex {
    expert_map: Arc<RwLock<HashMap<ExpertId, ExpertDefinition>>>,
    shard_map: Arc<RwLock<HashMap<ShardId, ExpertId>>>,
}

impl ShardIndex {
    pub fn new() -> Self {
        Self {
            expert_map: Arc::new(RwLock::new(HashMap::new())),
            shard_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_expert(&self, expert: Expert) -> AuriaResult<()> {
        let definition = ExpertDefinition {
            expert_id: expert.expert_id,
            shard_ids: expert.shards.clone(),
            tensor_layout: expert.tensor_layout,
        };

        let mut expert_map = self.expert_map.write().await;
        let mut shard_map = self.shard_map.write().await;

        // Validate that all shards are unique and not already assigned
        for shard_id in &expert.shards {
            if let Some(existing_expert) = shard_map.get(shard_id) {
                return Err(AuriaError::ExecutionError(format!(
                    "Shard {:?} already assigned to expert {:?}",
                    shard_id, existing_expert
                )));
            }
        }

        // Register expert and shard mappings
        expert_map.insert(expert.expert_id, definition);
        for shard_id in &expert.shards {
            shard_map.insert(*shard_id, expert.expert_id);
        }

        Ok(())
    }

    pub async fn get_expert_definition(&self, expert_id: ExpertId) -> AuriaResult<ExpertDefinition> {
        let expert_map = self.expert_map.read().await;
        expert_map.get(&expert_id)
            .cloned()
            .ok_or(AuriaError::ExpertNotFound(expert_id))
    }

    pub async fn get_expert_for_shard(&self, shard_id: ShardId) -> AuriaResult<ExpertId> {
        let shard_map = self.shard_map.read().await;
        shard_map.get(&shard_id)
            .copied()
            .ok_or(AuriaError::ShardNotFound(shard_id))
    }

    pub async fn get_shards_for_expert(&self, expert_id: ExpertId) -> AuriaResult<Vec<ShardId>> {
        let expert_map = self.expert_map.read().await;
        expert_map.get(&expert_id)
            .map(|def| def.shard_ids.clone())
            .ok_or(AuriaError::ExpertNotFound(expert_id))
    }

    pub async fn shard_exists(&self, shard_id: ShardId) -> bool {
        let shard_map = self.shard_map.read().await;
        shard_map.contains_key(&shard_id)
    }

    pub async fn expert_exists(&self, expert_id: ExpertId) -> bool {
        let expert_map = self.expert_map.read().await;
        expert_map.contains_key(&expert_id)
    }

    pub async fn get_all_experts(&self) -> Vec<ExpertId> {
        let expert_map = self.expert_map.read().await;
        expert_map.keys().copied().collect()
    }

    pub async fn get_all_shards(&self) -> Vec<ShardId> {
        let shard_map = self.shard_map.read().await;
        shard_map.keys().copied().collect()
    }

    pub async fn remove_expert(&self, expert_id: ExpertId) -> AuriaResult<()> {
        let mut expert_map = self.expert_map.write().await;
        let mut shard_map = self.shard_map.write().await;

        if let Some(definition) = expert_map.remove(&expert_id) {
            for shard_id in &definition.shard_ids {
                shard_map.remove(shard_id);
            }
            Ok(())
        } else {
            Err(AuriaError::ExpertNotFound(expert_id))
        }
    }

    pub async fn clear(&self) {
        let mut expert_map = self.expert_map.write().await;
        let mut shard_map = self.shard_map.write().await;
        expert_map.clear();
        shard_map.clear();
    }
}