// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Main expert assembly interface for AURIA Runtime Core.
//     Provides the primary entry point for expert assembly operations,
//     coordinating between shard index, expert assembler, and cache layers.

use super::{AuriaError, AuriaResult, Expert, ExpertId, Shard, ShardId, Tensor};
use super::expert_assembler::{AssemblyConfig, ExpertAssembler};
use super::expert_cache::ExpertCache;
use super::shard_index::ShardIndex;
use super::license_manager::LicenseManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyRequest {
    pub expert_id: ExpertId,
    pub force_reassembly: bool,
    pub priority: AssemblyPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssemblyPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyResponse {
    pub expert_id: ExpertId,
    pub tensor: Tensor,
    pub assembly_time_ms: u64,
    pub cache_hit: bool,
    pub source: AssemblySource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssemblySource {
    Cache,
    Assembly,
    Error,
}

pub struct ExpertAssemblyManager {
    shard_index: Arc<ShardIndex>,
    expert_assembler: Arc<ExpertAssembler>,
    expert_cache: Arc<ExpertCache>,
    license_manager: Arc<LicenseManager>,
    assembly_config: AssemblyConfig,
    stats: Arc<RwLock<AssemblyStats>>,
}

#[derive(Debug, Clone, Default)]
struct AssemblyStats {
    total_requests: u64,
    cache_hits: u64,
    successful_assemblies: u64,
    failed_assemblies: u64,
    total_assembly_time: u64,
    cache_size: usize,
}

impl ExpertAssemblyManager {
    pub fn new(
        shard_index: Arc<ShardIndex>,
        expert_assembler: Arc<ExpertAssembler>,
        expert_cache: Arc<ExpertCache>,
        license_manager: Arc<LicenseManager>,
        assembly_config: AssemblyConfig,
    ) -> Self {
        Self {
            shard_index,
            expert_assembler,
            expert_cache,
            license_manager,
            assembly_config,
            stats: Arc::new(RwLock::new(AssemblyStats::default())),
        }
    }

    pub async fn assemble_expert(
        &self,
        request: AssemblyRequest,
    ) -> AuriaResult<AssemblyResponse> {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Check cache first
        if !request.force_reassembly {
            if let Some(tensor) = self.expert_cache.get_expert(request.expert_id).await? {
                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.cache_hits += 1;
                    stats.cache_size = self.expert_cache.get_cache_stats().await.vram_count;
                }

                return Ok(AssemblyResponse {
                    expert_id: request.expert_id,
                    tensor,
                    assembly_time_ms: 0,
                    cache_hit: true,
                    source: AssemblySource::Cache,
                });
            }
        }

        // Get expert definition
        let definition = self.shard_index.get_expert_definition(request.expert_id).await?;

        // Assemble expert
        let start_time = std::time::Instant::now();
        let tensor = self.expert_assembler.assemble_expert(request.expert_id).await?;
        let assembly_time = start_time.elapsed().as_millis() as u64;

        // Store in cache
        self.expert_cache.store_expert(request.expert_id, tensor.clone(), Tier::Nano).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.successful_assemblies += 1;
            stats.total_assembly_time += assembly_time;
            stats.cache_size = self.expert_cache.get_cache_stats().await.vram_count;
        }

        Ok(AssemblyResponse {
            expert_id: request.expert_id,
            tensor,
            assembly_time_ms: assembly_time,
            cache_hit: false,
            source: AssemblySource::Assembly,
        })
    }

    pub async fn assemble_multiple_experts(
        &self,
        requests: Vec<AssemblyRequest>,
    ) -> Vec<AuriaResult<AssemblyResponse>> {
        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            let result = self.assemble_expert(request).await;
            results.push(result);
        }

        results
    }

    pub async fn register_expert(
        &self,
        expert: Expert,
    ) -> AuriaResult<()> {
        // Register with shard index
        self.shard_index.register_expert(expert.clone()).await?;

        // Pre-assemble and cache if configured
        if self.assembly_config.deterministic_assembly {
            self.assemble_expert(AssemblyRequest {
                expert_id: expert.expert_id,
                force_reassembly: false,
                priority: AssemblyPriority::Normal,
            }).await?;
        }

        Ok(())
    }

    pub async fn register_multiple_experts(
        &self,
        experts: Vec<Expert>,
    ) -> Vec<AuriaResult<()>> {
        let mut results = Vec::with_capacity(experts.len());

        for expert in experts {
            let result = self.register_expert(expert).await;
            results.push(result);
        }

        results
    }

    pub async fn get_assembly_stats(&self) -> AssemblyStats {
        self.stats.read().await.clone()
    }

    pub async fn clear_cache(&self) {
        self.expert_cache.clear_cache().await;
    }

    pub async fn cleanup_expired_entries(&self) {
        self.expert_cache.cleanup_expired_entries().await;
    }

    pub async fn get_cache_stats(&self) -> super::expert_cache::CacheStats {
        self.expert_cache.get_cache_stats().await
    }

    pub async fn validate_assembly(
        &self,
        expert_id: ExpertId,
    ) -> AuriaResult<bool> {
        // Get expert definition
        let definition = self.shard_index.get_expert_definition(expert_id).await?;

        // Retrieve shards
        let shard_ids = definition.shard_ids;
        let shards = self.expert_assembler.retrieve_and_validate_shards(&shard_ids).await?;

        // Get cached tensor if available
        let tensor = if let Some(cached_tensor) = self.expert_cache.get_expert(expert_id).await? {
            cached_tensor
        } else {
            // If not cached, assemble it
            self.expert_assembler.assemble_tensor(&definition, &shards).await?
        };

        // Verify assembly
        self.expert_assembler.verify_assembly(&definition, &shards, &tensor).await?;

        Ok(true)
    }

    pub async fn get_registered_experts(&self) -> Vec<ExpertId> {
        self.shard_index.get_all_experts().await
    }

    pub async fn get_registered_shards(&self) -> Vec<ShardId> {
        self.shard_index.get_all_shards().await
    }
}

impl ExpertAssemblyManager {
    // Helper methods for testing
    #[cfg(test)]
    pub async fn get_mock_assembly_manager() -> Self {
        let shard_index = Arc::new(ShardIndex::new());
        let license_manager = Arc::new(LicenseManager::new(Default::default()));
        let expert_assembler = Arc::new(ExpertAssembler::new(
            shard_index.clone(),
            license_manager.clone(),
            AssemblyConfig::default(),
        ));
        let expert_cache = Arc::new(ExpertCache::new(Default::default()));

        Self::new(
            shard_index,
            expert_assembler,
            expert_cache,
            license_manager,
            AssemblyConfig::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_expert_assembly_manager_basic_flow() {
        let manager = ExpertAssemblyManager::get_mock_assembly_manager().await;

        // Create a mock expert
        let expert_id = ExpertId([1; 32]);
        let shard1 = ShardId([1; 32]);
        let shard2 = ShardId([2; 32]);

        let expert = Expert {
            expert_id,
            shards: vec![shard1, shard2],
            tensor_layout: TensorLayout {
                offset: 0,
                stride: 1,
                shape: vec![4, 4],
            },
        };

        // Register expert
        manager.register_expert(expert).await.unwrap();

        // Assemble expert
        let request = AssemblyRequest {
            expert_id,
            force_reassembly: false,
            priority: AssemblyPriority::Normal,
        };

        let response = manager.assemble_expert(request).await.unwrap();
        assert_eq!(response.expert_id, expert_id);
        assert!(!response.cache_hit);
        assert!(response.assembly_time_ms > 0);
        assert_eq!(response.source, AssemblySource::Assembly);

        // Assemble again (should hit cache)
        let cached_response = manager.assemble_expert(request).await.unwrap();
        assert_eq!(cached_response.expert_id, expert_id);
        assert!(cached_response.cache_hit);
        assert_eq!(cached_response.source, AssemblySource::Cache);
    }

    #[tokio::test]
    async fn test_expert_assembly_manager_multiple_experts() {
        let manager = ExpertAssemblyManager::get_mock_assembly_manager().await;

        // Create multiple mock experts
        let expert1 = Expert {
            expert_id: ExpertId([1; 32]),
            shards: vec![ShardId([1; 32])],
            tensor_layout: TensorLayout {
                offset: 0,
                stride: 1,
                shape: vec![2, 2],
            },
        };

        let expert2 = Expert {
            expert_id: ExpertId([2; 32]),
            shards: vec![ShardId([2; 32])],
            tensor_layout: TensorLayout {
                offset: 0,
                stride: 1,
                shape: vec![2, 2],
            },
        };

        // Register experts
        manager.register_multiple_experts(vec![expert1, expert2]).await;

        // Assemble multiple experts
        let requests = vec![
            AssemblyRequest {
                expert_id: ExpertId([1; 32]),
                force_reassembly: false,
                priority: AssemblyPriority::Normal,
            },
            AssemblyRequest {
                expert_id: ExpertId([2; 32]),
                force_reassembly: false,
                priority: AssemblyPriority::Normal,
            },
        ];

        let responses = manager.assemble_multiple_experts(requests).await;
        assert_eq!(responses.len(), 2);
        assert!(responses.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn test_expert_assembly_manager_stats() {
        let manager = ExpertAssemblyManager::get_mock_assembly_manager().await;

        // Create a mock expert
        let expert = Expert {
            expert_id: ExpertId([1; 32]),
            shards: vec![ShardId([1; 32])],
            tensor_layout: TensorLayout {
                offset: 0,
                stride: 1,
                shape: vec![2, 2],
            },
        };

        // Register expert
        manager.register_expert(expert).await.unwrap();

        // Assemble expert twice
        let request = AssemblyRequest {
            expert_id: ExpertId([1; 32]),
            force_reassembly: false,
            priority: AssemblyPriority::Normal,
        };

        manager.assemble_expert(request.clone()).await.unwrap();
        manager.assemble_expert(request.clone()).await.unwrap();

        // Get stats
        let stats = manager.get_assembly_stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1); // Second call hits cache
        assert_eq!(stats.successful_assemblies, 1);
        assert_eq!(stats.failed_assemblies, 0);
        assert!(stats.total_assembly_time > 0);
    }

    #[tokio::test]
    async fn test_expert_assembly_manager_validation() {
        let manager = ExpertAssemblyManager::get_mock_assembly_manager().await;

        // Create a mock expert
        let expert = Expert {
            expert_id: ExpertId([1; 32]),
            shards: vec![ShardId([1; 32])],
            tensor_layout: TensorLayout {
                offset: 0,
                stride: 1,
                shape: vec![2, 2],
            },
        };

        // Register expert
        manager.register_expert(expert).await.unwrap();

        // Validate assembly
        let valid = manager.validate_assembly(ExpertId([1; 32])).await.unwrap();
        assert!(valid);
    }
}
