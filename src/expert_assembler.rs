// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Expert Assembler for AURIA Runtime Core.
//     Implements deterministic assembly of experts from licensed shards,
//     ensuring cryptographically verifiable and reproducible expert construction.

use super::{AuriaError, AuriaResult, Expert, ExpertId, Shard, ShardId, Tensor, TensorDType};
use super::shard_index::ShardIndex;
use super::license_manager::LicenseManager;
use super::storage::StorageTier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyConfig {
    pub assembly_strategy: AssemblyStrategy,
    pub deterministic_assembly: bool,
    pub verify_assembly: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssemblyStrategy {
    Concat,
    WeightedSum {
        weights: HashMap<ShardId, f32>,
    },
    Custom {
        function: String,
    },
}

impl Default for AssemblyConfig {
    fn default() -> Self {
        Self {
            assembly_strategy: AssemblyStrategy::Concat,
            deterministic_assembly: true,
            verify_assembly: true,
        }
    }
}

pub struct ExpertAssembler {
    shard_index: Arc<ShardIndex>,
    license_manager: Arc<LicenseManager>,
    config: AssemblyConfig,
    assembly_cache: Arc<RwLock<HashMap<ExpertId, AssemblyRecord>>>,
}

#[derive(Debug, Clone)]
struct AssemblyRecord {
    tensor: Tensor,
    assembly_hash: String,
    timestamp: u64,
    shard_ids: Vec<ShardId>,
}

impl ExpertAssembler {
    pub fn new(
        shard_index: Arc<ShardIndex>,
        license_manager: Arc<LicenseManager>,
        config: AssemblyConfig,
    ) -> Self {
        Self {
            shard_index,
            license_manager,
            config,
            assembly_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn assemble_expert(
        &self,
        expert_id: ExpertId,
    ) -> AuriaResult<Tensor> {
        // Check cache first
        if let Some(cached) = self.assembly_cache.read().await.get(&expert_id) {
            return Ok(cached.tensor.clone());
        }

        // Get expert definition
        let definition = self.shard_index.get_expert_definition(expert_id).await?;

        // Retrieve and validate shards
        let shards = self.retrieve_and_validate_shards(&definition.shard_ids).await?;

        // Assemble tensor
        let tensor = self.assemble_tensor(&definition, &shards).await?;

        // Verify assembly if required
        if self.config.verify_assembly {
            self.verify_assembly(&definition, &shards, &tensor).await?;
        }

        // Cache the result
        let assembly_record = AssemblyRecord {
            tensor: tensor.clone(),
            assembly_hash: self.calculate_assembly_hash(&definition, &shards).await,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            shard_ids: definition.shard_ids.clone(),
        };

        self.assembly_cache.write().await.insert(expert_id, assembly_record);

        Ok(tensor)
    }

    async fn retrieve_and_validate_shards(
        &self,
        shard_ids: &[ShardId],
    ) -> AuriaResult<Vec<Shard>> {
        let mut shards = Vec::with_capacity(shard_ids.len());

        for shard_id in shard_ids {
            // Retrieve shard from storage
            let shard = self.retrieve_shard(*shard_id).await?;

            // Validate license
            self.validate_shard_license(*shard_id).await?;

            shards.push(shard);
        }

        Ok(shards)
    }

    async fn retrieve_shard(
        &self,
        shard_id: ShardId,
    ) -> AuriaResult<Shard> {
        // This would normally call the storage layer to retrieve the shard
        // For now, we'll return a placeholder implementation
        // In a real implementation, this would load from disk/network storage

        // Check if we have this shard in a mock storage
        let mock_shard = self.get_mock_shard(shard_id);
        if let Some(shard) = mock_shard {
            return Ok(shard);
        }

        Err(AuriaError::ShardNotFound(shard_id))
    }

    async fn validate_shard_license(
        &self,
        shard_id: ShardId,
    ) -> AuriaResult<()> {
        // Check license validity
        if !self.license_manager.is_license_valid_for_shard(shard_id).await? {
            return Err(AuriaError::LicenseInvalid(shard_id));
        }
        Ok(())
    }

    async fn assemble_tensor(
        &self,
        definition: &super::ExpertDefinition,
        shards: &[Shard],
    ) -> AuriaResult<Tensor> {
        match &self.config.assembly_strategy {
            AssemblyStrategy::Concat => self.concat_assembly(definition, shards).await,
            AssemblyStrategy::WeightedSum { weights } => self.weighted_sum_assembly(definition, shards, weights).await,
            AssemblyStrategy::Custom { function } => {
                Err(AuriaError::ExecutionError(format!(
                    "Custom assembly function '{}' not implemented",
                    function
                )))
            }
        }
    }

    async fn concat_assembly(
        &self,
        definition: &super::ExpertDefinition,
        shards: &[Shard],
    ) -> AuriaResult<Tensor> {
        // Simple concatenation strategy
        let mut data = Vec::new();
        let mut shape = definition.tensor_layout.shape.clone();

        // Calculate total size
        let mut total_size: usize = 0;
        for shard in shards {
            total_size += shard.tensor.data.len();
        }

        // Concatenate data
        for shard in shards {
            data.extend_from_slice(&shard.tensor.data);
        }

        // Update shape based on tensor layout
        shape[0] = total_size as u32 / 4; // Assuming FP16 (2 bytes) * 2 for simplicity

        Ok(Tensor {
            data,
            shape,
            dtype: TensorDType::FP16,
        })
    }

    async fn weighted_sum_assembly(
        &self,
        definition: &super::ExpertDefinition,
        shards: &[Shard],
        weights: &HashMap<ShardId, f32>,
    ) -> AuriaResult<Tensor> {
        // Weighted sum strategy
        if shards.len() != weights.len() {
            return Err(AuriaError::ExecutionError(
                "Number of shards must match number of weights".to_string()
            ));
        }

        // Convert shards to f32 for weighted operations
        let mut weighted_tensors = Vec::new();
        for shard in shards {
            let f32_data = self.convert_to_f32(&shard.tensor.data)?;
            let weight = *weights.get(&shard.shard_id).unwrap_or(&1.0);
            let weighted_data: Vec<f32> = f32_data.iter().map(|v| v * weight).collect();
            weighted_tensors.push(weighted_data);
        }

        // Sum all weighted tensors
        let mut result = vec![0.0; weighted_tensors[0].len()];
        for wt in weighted_tensors {
            for i in 0..wt.len() {
                result[i] += wt[i];
            }
        }

        // Convert back to bytes
        let bytes: Vec<u8> = result.iter().flat_map(|f| f.to_le_bytes()).collect();

        Ok(Tensor {
            data: bytes,
            shape: definition.tensor_layout.shape.clone(),
            dtype: TensorDType::FP16,
        })
    }

    async fn verify_assembly(
        &self,
        definition: &super::ExpertDefinition,
        shards: &[Shard],
        tensor: &Tensor,
    ) -> AuriaResult<()> {
        // Verify that the assembly is deterministic
        let assembly_hash = self.calculate_assembly_hash(definition, shards).await;
        let tensor_hash = self.calculate_tensor_hash(tensor);

        if assembly_hash != tensor_hash {
            return Err(AuriaError::ExecutionError(
                "Assembly verification failed: hashes do not match".to_string()
            ));
        }

        Ok(())
    }

    async fn calculate_assembly_hash(
        &self,
        definition: &super::ExpertDefinition,
        shards: &[Shard],
    ) -> String {
        use sha2::Sha256;

        let mut hasher = Sha256::new();

        // Hash expert ID
        hasher.update(&definition.expert_id.0);

        // Hash shard IDs in sorted order for determinism
        let mut sorted_shard_ids: Vec<&ShardId>> = shards.iter().map(|s| &s.shard_id).collect();
        sorted_shard_ids.sort();
        for shard_id in sorted_shard_ids {
            hasher.update(&shard_id.0);
        }

        // Hash shard data hashes
        for shard in shards {
            let shard_hash = self.calculate_tensor_hash(&shard.tensor);
            hasher.update(shard_hash.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    fn calculate_tensor_hash(
        &self,
        tensor: &Tensor,
    ) -> String {
        use sha2::Sha256;

        let mut hasher = Sha256::new();
        hasher.update(&tensor.data);
        format!("{:x}", hasher.finalize())
    }

    fn get_mock_shard(
        &self,
        shard_id: ShardId,
    ) -> Option<Shard> {
        // Mock implementation - in a real system this would load from storage
        // Create a simple mock shard based on the shard ID
        let mut data = Vec::new();
        for i in 0..32 {
            data.push((shard_id.0[i % 32] ^ i) as u8);
        }

        Some(Shard {
            shard_id,
            expert_id: ExpertId([0; 32]), // Placeholder
            tensor: Tensor {
                data,
                shape: vec![4, 4, 4],
                dtype: TensorDType::FP16,
            },
            metadata: super::ShardMetadata {
                owner: super::PublicKey([0; 32]),
                license_hash: None,
                created_at: 0,
                version: 1,
            },
        })
    }

    fn convert_to_f32(
        &self,
        data: &[ u8 ],
    ) -> AuriaResult<Vec<f32>> {
        if data.len() % 4 != 0 {
            return Err(AuriaError::ExecutionError(
                "Data length must be multiple of 4 for FP32 conversion".to_string()
            ));
        }

        let mut result = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks(4) {
            let bytes = [
                chunk[0], chunk[1], chunk[2], chunk[3]
            ];
            let f = f32::from_le_bytes(bytes);
            result.push(f);
        }

        Ok(result)
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.assembly_cache.write().await;
        cache.clear();
    }

    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.assembly_cache.read().await;
        (cache.len(), cache.capacity().unwrap_or(0))
    }
}