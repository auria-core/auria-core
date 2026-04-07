// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Expert Cache for AURIA Runtime Core.
//     Implements a multi-level cache hierarchy for assembled expert tensors,
//     optimizing for performance and memory usage across different hardware tiers.

use super::{AuriaError, AuriaResult, ExpertId, Tensor, Tier};
use super::expert_assembler::AssemblyConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub vram_cache_size: usize,
    pub ram_cache_size: usize,
    pub disk_cache_size: usize,
    pub eviction_policy: EvictionPolicy,
    pub cache_ttl_seconds: u64,
    pub enable_vram_cache: bool,
    pub enable_ram_cache: bool,
    pub enable_disk_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    LruK(u32),
    Random,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            vram_cache_size: 32,
            ram_cache_size: 512,
            disk_cache_size: 1024,
            eviction_policy: EvictionPolicy::Lru,
            cache_ttl_seconds: 3600,
            enable_vram_cache: true,
            enable_ram_cache: true,
            enable_disk_cache: false, // Disk cache disabled by default
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    tensor: Tensor,
    last_used: u64,
    creation_time: u64,
    access_count: u64,
    tier: Tier,
}

pub struct ExpertCache {
    config: CacheConfig,
    vram_cache: Arc<RwLock<HashMap<ExpertId, CacheEntry>>>,
    ram_cache: Arc<RwLock<HashMap<ExpertId, CacheEntry>>>,
    disk_cache: Arc<RwLock<HashMap<ExpertId, CacheEntry>>>,
    vram_order: Arc<RwLock<Vec<ExpertId>>>,
    ram_order: Arc<RwLock<Vec<ExpertId>>>,
    disk_order: Arc<RwLock<Vec<ExpertId>>>,
}

impl ExpertCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            vram_cache: Arc::new(RwLock::new(HashMap::new())),
            ram_cache: Arc::new(RwLock::new(HashMap::new())),
            disk_cache: Arc::new(RwLock::new(HashMap::new())),
            vram_order: Arc::new(RwLock::new(Vec::new())),
            ram_order: Arc::new(RwLock::new(Vec::new())),
            disk_order: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn get_expert(
        &self,
        expert_id: ExpertId,
    ) -> AuriaResult<Option<Tensor>> {
        // Check VRAM cache first
        if self.config.enable_vram_cache {
            if let Some(entry) = self.vram_cache.read().await.get(&expert_id) {
                self.update_access(&entry.tier, expert_id).await;
                return Ok(Some(entry.tensor.clone()));
            }
        }

        // Check RAM cache
        if self.config.enable_ram_cache {
            if let Some(entry) = self.ram_cache.read().await.get(&expert_id) {
                self.update_access(&entry.tier, expert_id).await;
                return Ok(Some(entry.tensor.clone()));
            }
        }

        // Check disk cache
        if self.config.enable_disk_cache {
            if let Some(entry) = self.disk_cache.read().await.get(&expert_id) {
                self.update_access(&entry.tier, expert_id).await;
                return Ok(Some(entry.tensor.clone()));
            }
        }

        Ok(None)
    }

    pub async fn store_expert(
        &self,
        expert_id: ExpertId,
        tensor: Tensor,
        tier: Tier,
    ) -> AuriaResult<()> {
        let entry = CacheEntry {
            tensor: tensor.clone(),
            last_used: current_timestamp(),
            creation_time: current_timestamp(),
            access_count: 1,
            tier,
        };

        // Determine appropriate cache level
        let cache_level = self.get_cache_level(tier);

        match cache_level {
            CacheLevel::Vram => {
                if self.config.enable_vram_cache {
                    self.store_in_vram(expert_id, entry).await?;
                }
            }
            CacheLevel::Ram => {
                if self.config.enable_ram_cache {
                    self.store_in_ram(expert_id, entry).await?;
                }
            }
            CacheLevel::Disk => {
                if self.config.enable_disk_cache {
                    self.store_in_disk(expert_id, entry).await?;
                }
            }
        }

        Ok(())
    }

    pub async fn remove_expert(
        &self,
        expert_id: ExpertId,
    ) -> AuriaResult<()> {
        // Remove from all cache levels
        if self.config.enable_vram_cache {
            self.vram_cache.write().await.remove(&expert_id);
            self.remove_from_order(&self.vram_order, expert_id).await;
        }

        if self.config.enable_ram_cache {
            self.ram_cache.write().await.remove(&expert_id);
            self.remove_from_order(&self.ram_order, expert_id).await;
        }

        if self.config.enable_disk_cache {
            self.disk_cache.write().await.remove(&expert_id);
            self.remove_from_order(&self.disk_order, expert_id).await;
        }

        Ok(())
    }

    pub async fn clear_cache(&self) {
        if self.config.enable_vram_cache {
            self.vram_cache.write().await.clear();
            self.vram_order.write().await.clear();
        }

        if self.config.enable_ram_cache {
            self.ram_cache.write().await.clear();
            self.ram_order.write().await.clear();
        }

        if self.config.enable_disk_cache {
            self.disk_cache.write().await.clear();
            self.disk_order.write().await.clear();
        }
    }

    pub async fn get_cache_stats(&self) -> CacheStats {
        let vram_count = if self.config.enable_vram_cache {
            self.vram_cache.read().await.len()
        } else {
            0
        };

        let ram_count = if self.config.enable_ram_cache {
            self.ram_cache.read().await.len()
        } else {
            0
        };

        let disk_count = if self.config.enable_disk_cache {
            self.disk_cache.read().await.len()
        } else {
            0
        };

        CacheStats {
            vram_count,
            ram_count,
            disk_count,
            total_size_bytes: 0, // Would need to calculate actual size
            hit_count: 0, // Would need to track hits
            miss_count: 0, // Would need to track misses
        }
    }

    async fn store_in_vram(
        &self,
        expert_id: ExpertId,
        entry: CacheEntry,
    ) -> AuriaResult<()> {
        let mut vram_cache = self.vram_cache.write().await;
        let mut vram_order = self.vram_order.write().await;

        // Evict if necessary
        if vram_cache.len() >= self.config.vram_cache_size {
            self.evict_from_cache(
                CacheLevel::Vram,
                self.config.vram_cache_size,
                &mut vram_cache,
                &mut vram_order,
            ).await?;
        }

        vram_cache.insert(expert_id, entry);
        vram_order.push(expert_id);

        Ok(())
    }

    async fn store_in_ram(
        &self,
        expert_id: ExpertId,
        entry: CacheEntry,
    ) -> AuriaResult<()> {
        let mut ram_cache = self.ram_cache.write().await;
        let mut ram_order = self.ram_order.write().await;

        // Evict if necessary
        if ram_cache.len() >= self.config.ram_cache_size {
            self.evict_from_cache(
                CacheLevel::Ram,
                self.config.ram_cache_size,
                &mut ram_cache,
                &mut ram_order,
            ).await?;
        }

        ram_cache.insert(expert_id, entry);
        ram_order.push(expert_id);

        Ok(())
    }

    async fn store_in_disk(
        &self,
        expert_id: ExpertId,
        entry: CacheEntry,
    ) -> AuriaResult<()> {
        let mut disk_cache = self.disk_cache.write().await;
        let mut disk_order = self.disk_order.write().await;

        // Evict if necessary
        if disk_cache.len() >= self.config.disk_cache_size {
            self.evict_from_cache(
                CacheLevel::Disk,
                self.config.disk_cache_size,
                &mut disk_cache,
                &mut disk_order,
            ).await?;
        }

        disk_cache.insert(expert_id, entry);
        disk_order.push(expert_id);

        Ok(())
    }

    async fn evict_from_cache(
        &self,
        level: CacheLevel,
        max_size: usize,
        cache: &mut HashMap<ExpertId, CacheEntry>,
        order: &mut Vec<ExpertId>,
    ) -> AuriaResult<()> {
        match self.config.eviction_policy {
            EvictionPolicy::Lru => {
                // Remove least recently used
                while cache.len() >= max_size {
                    if let Some(oldest) = order.first().cloned() {
                        cache.remove(&oldest);
                        order.remove(0);
                    }
                }
            }
            EvictionPolicy::Lfu => {
                // Remove least frequently used (simplified)
                let mut entries: Vec<(_)>> = cache.iter().map(|(&id, entry)| (id, entry)).collect();
                entries.sort_by_key(|(_, entry)| entry.access_count);

                while cache.len() >= max_size && !entries.is_empty() {
                    if let Some((&oldest, _)) = entries.first() {
                        cache.remove(oldest);
                        self.remove_from_order(order, *oldest).await;
                        entries.remove(0);
                    }
                }
            }
            EvictionPolicy::LruK(k) => {
                // Simplified LRU-K: remove based on last used time
                let mut entries: Vec<(_)>> = cache.iter().map(|(&id, entry)| (id, entry)).collect();
                entries.sort_by_key(|(_, entry)| entry.last_used);

                while cache.len() >= max_size && !entries.is_empty() {
                    if let Some((&oldest, _)) = entries.first() {
                        cache.remove(oldest);
                        self.remove_from_order(order, *oldest).await;
                        entries.remove(0);
                    }
                }
            }
            EvictionPolicy::Random => {
                // Remove random entries
                while cache.len() >= max_size {
                    if let Some(idx) = rand::random::<usize>() % cache.len() {
                        let keys: Vec<ExpertId>> = cache.keys().cloned().collect();
                        if let Some(&key) = keys.get(idx) {
                            cache.remove(&key);
                            self.remove_from_order(order, key).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn update_access(
        &self,
        tier: &Tier,
        expert_id: ExpertId,
    ) {
        // Update access time and count
        let timestamp = current_timestamp();

        match tier {
            Tier::Nano | Tier::Standard => {
                if self.config.enable_vram_cache {
                    if let Some(mut entry) = self.vram_cache.write().await.get_mut(&expert_id) {
                        entry.last_used = timestamp;
                        entry.access_count += 1;
                    }
                }
            }
            Tier::Pro | Tier::Max => {
                if self.config.enable_ram_cache {
                    if let Some(mut entry) = self.ram_cache.write().await.get_mut(&expert_id) {
                        entry.last_used = timestamp;
                        entry.access_count += 1;
                    }
                }
            }
        }
    }

    async fn remove_from_order(
        &self,
        order: &Arc<RwLock<Vec<ExpertId>>>,
        expert_id: ExpertId,
    ) {
        let mut order = order.write().await;
        if let Some(pos) = order.iter().position(|id| *id == expert_id) {
            order.remove(pos);
        }
    }

    fn get_cache_level(
        &self,
        tier: Tier,
    ) -> CacheLevel {
        match tier {
            Tier::Nano | Tier::Standard => CacheLevel::Vram,
            Tier::Pro | Tier::Max => CacheLevel::Ram,
        }
    }

    pub async fn cleanup_expired_entries(&self) {
        let ttl = self.config.cache_ttl_seconds;
        let now = current_timestamp();

        if self.config.enable_vram_cache {
            let mut vram_cache = self.vram_cache.write().await;
            let mut vram_order = self.vram_order.write().await;
            self.cleanup_expired_in_cache(&mut vram_cache, &mut vram_order, ttl, now).await;
        }

        if self.config.enable_ram_cache {
            let mut ram_cache = self.ram_cache.write().await;
            let mut ram_order = self.ram_order.write().await;
            self.cleanup_expired_in_cache(&mut ram_cache, &mut ram_order, ttl, now).await;
        }

        if self.config.enable_disk_cache {
            let mut disk_cache = self.disk_cache.write().await;
            let mut disk_order = self.disk_order.write().await;
            self.cleanup_expired_in_cache(&mut disk_cache, &mut disk_order, ttl, now).await;
        }
    }

    async fn cleanup_expired_in_cache(
        &self,
        cache: &mut HashMap<ExpertId, CacheEntry>,
        order: &mut Vec<ExpertId>,
        ttl: u64,
        now: u64,
    ) {
        let expired_threshold = now - ttl;
        let mut expired_ids = Vec::new();

        for (&expert_id, entry) in cache.iter() {
            if entry.creation_time < expired_threshold {
                expired_ids.push(expert_id);
            }
        }

        for expert_id in expired_ids {
            cache.remove(&expert_id);
            if let Some(pos) = order.iter().position(|id| *id == expert_id) {
                order.remove(pos);
            }
        }
    }
}

#[derive(Debug, Clone)]
enum CacheLevel {
    Vram,
    Ram,
    Disk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub vram_count: usize,
    pub ram_count: usize,
    pub disk_count: usize,
    pub total_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_expert_cache_basic_operations() {
        let config = CacheConfig::default();
        let cache = ExpertCache::new(config);

        let expert_id = ExpertId([1; 32]);
        let tensor = Tensor {
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: TensorDType::FP16,
        };

        // Store expert
        cache.store_expert(expert_id, tensor.clone(), Tier::Nano).await.unwrap();

        // Retrieve expert
        let retrieved = cache.get_expert(expert_id).await.unwrap();
        assert_eq!(retrieved, Some(tensor));

        // Remove expert
        cache.remove_expert(expert_id).await.unwrap();
        let after_remove = cache.get_expert(expert_id).await.unwrap();
        assert_eq!(after_remove, None);
    }

    #[tokio::test]
    async fn test_expert_cache_eviction() {
        let mut config = CacheConfig::default();
        config.vram_cache_size = 2; // Small cache for testing
        let cache = ExpertCache::new(config);

        // Add three experts (should evict one)
        let expert1 = ExpertId([1; 32]);
        let expert2 = ExpertId([2; 32]);
        let expert3 = ExpertId([3; 32]);

        cache.store_expert(expert1, Tensor::default(), Tier::Nano).await.unwrap();
        cache.store_expert(expert2, Tensor::default(), Tier::Nano).await.unwrap();
        cache.store_expert(expert3, Tensor::default(), Tier::Nano).await.unwrap();

        // Should have evicted one
        let stats = cache.get_cache_stats().await;
        assert_eq!(stats.vram_count, 2);
    }

    #[tokio::test]
    async fn test_expert_cache_tier_aware() {
        let config = CacheConfig::default();
        let cache = ExpertCache::new(config);

        let nano_expert = ExpertId([1; 32]);
        let pro_expert = ExpertId([2; 32]);

        // Nano tier should go to VRAM cache
        cache.store_expert(nano_expert, Tensor::default(), Tier::Nano).await.unwrap();

        // Pro tier should go to RAM cache
        cache.store_expert(pro_expert, Tensor::default(), Tier::Pro).await.unwrap();

        // Verify placement
        let stats = cache.get_cache_stats().await;
        assert_eq!(stats.vram_count, 1);
        assert_eq!(stats.ram_count, 1);
    }
}
