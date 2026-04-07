use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageTier {
    Vram,
    Ram,
    Disk,
    Network,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub vram_cache_size_mb: u32,
    pub ram_cache_size_mb: u32,
    pub disk_cache_path: String,
    pub disk_cache_size_mb: u32,
    pub network_cache_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCacheEntry {
    pub shard_id: String,
    pub data: Vec<u8>,
    pub tier: StorageTier,
    pub size_bytes: u64,
    pub last_accessed: Instant,
    pub hit_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardStorageStats {
    pub vram_count: usize,
    pub vram_size_mb: u32,
    pub ram_count: usize,
    pub ram_size_mb: u32,
    pub disk_count: usize,
    pub disk_size_mb: u32,
    pub network_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardStorageError {
    pub shard_id: String,
    pub error_type: String,
    pub message: String,
}

pub struct ModelStore {
    config: StorageConfig,
    cache: Arc<Mutex<CacheManager>>,
    disk_path: String,
}

impl ModelStore {
    pub fn new(config: StorageConfig) -> AuriaResult<Self> {
        let disk_path = config.disk_cache_path.clone();

        // Create disk cache directory if it doesn't exist
        let path = Path::new(&disk_path);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }

        Ok(Self {
            config,
            cache: Arc::new(Mutex::new(CacheManager::new(config.clone()))),
            disk_path,
        })
    }

    pub fn load_shard(&self, shard_id: &str) -> AuriaResult<Vec<u8>> {
        let mut cache = self.cache.lock().unwrap();

        // Check VRAM cache first
        if let Some(entry) = cache.vram_cache.get(shard_id) {
            entry.last_accessed = Instant::now();
            entry.hit_count += 1;
            return Ok(entry.data.clone());
        }

        // Check RAM cache
        if let Some(entry) = cache.ram_cache.get(shard_id) {
            entry.last_accessed = Instant::now();
            entry.hit_count += 1;

            // Promote to VRAM if space available
            if cache.vram_cache_size < self.config.vram_cache_size_mb {
                let data = entry.data.clone();
                cache.vram_cache.insert(
                    shard_id.to_string(),
                    ShardCacheEntry {
                        shard_id: shard_id.to_string(),
                        data: data.clone(),
                        tier: StorageTier::Vram,
                        size_bytes: data.len() as u64,
                        last_accessed: Instant::now(),
                        hit_count: entry.hit_count,
                    }
                );
                cache.vram_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
            }

            return Ok(entry.data.clone());
        }

        // Check disk cache
        let disk_path = format!("{}/{}.shard", self.disk_path, shard_id);
        if Path::new(&disk_path).exists() {
            let data = std::fs::read(disk_path)?;

            // Add to RAM cache
            if cache.ram_cache_size < self.config.ram_cache_size_mb {
                cache.ram_cache.insert(
                    shard_id.to_string(),
                    ShardCacheEntry {
                        shard_id: shard_id.to_string(),
                        data: data.clone(),
                        tier: StorageTier::Ram,
                        size_bytes: data.len() as u64,
                        last_accessed: Instant::now(),
                        hit_count: 1,
                    }
                );
                cache.ram_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
            }

            return Ok(data);
        }

        // Check network storage (simulated)
        if self.config.network_cache_enabled {
            // Simulate network fetch
            let data = self.fetch_from_network(shard_id)?;

            // Store in disk cache
            self.store_in_disk(shard_id, &data)?;

            // Add to RAM cache
            if cache.ram_cache_size < self.config.ram_cache_size_mb {
                cache.ram_cache.insert(
                    shard_id.to_string(),
                    ShardCacheEntry {
                        shard_id: shard_id.to_string(),
                        data: data.clone(),
                        tier: StorageTier::Ram,
                        size_bytes: data.len() as u64,
                        last_accessed: Instant::now(),
                        hit_count: 1,
                    }
                );
                cache.ram_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
            }

            return Ok(data);
        }

        Err(AuriaError::StorageError(format!("Shard {} not found in any storage tier", shard_id)))
    }

    pub fn store_shard(&self, shard_id: &str, data: &[u8]) -> AuriaResult<()> {
        let mut cache = self.cache.lock().unwrap();

        // Store in VRAM cache if space available
        if cache.vram_cache_size < self.config.vram_cache_size_mb {
            cache.vram_cache.insert(
                shard_id.to_string(),
                ShardCacheEntry {
                    shard_id: shard_id.to_string(),
                    data: data.to_vec(),
                    tier: StorageTier::Vram,
                    size_bytes: data.len() as u64,
                    last_accessed: Instant::now(),
                    hit_count: 1,
                }
            );
            cache.vram_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
            return Ok(());
        }

        // Store in RAM cache if space available
        if cache.ram_cache_size < self.config.ram_cache_size_mb {
            cache.ram_cache.insert(
                shard_id.to_string(),
                ShardCacheEntry {
                    shard_id: shard_id.to_string(),
                    data: data.to_vec(),
                    tier: StorageTier::Ram,
                    size_bytes: data.len() as u64,
                    last_accessed: Instant::now(),
                    hit_count: 1,
                }
            );
            cache.ram_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
            return Ok(());
        }

        // Store in disk cache
        self.store_in_disk(shard_id, data)?;

        Ok(())
    }

    pub fn shard_exists(&self, shard_id: &str) -> bool {
        let cache = self.cache.lock().unwrap();

        // Check VRAM cache
        if cache.vram_cache.contains_key(shard_id) {
            return true;
        }

        // Check RAM cache
        if cache.ram_cache.contains_key(shard_id) {
            return true;
        }

        // Check disk cache
        let disk_path = format!("{}/{}.shard", self.disk_path, shard_id);
        if Path::new(&disk_path).exists() {
            return true;
        }

        // Check network storage (simulated)
        if self.config.network_cache_enabled {
            // Simulate network check
            return self.shard_exists_in_network(shard_id);
        }

        false
    }

    pub fn get_storage_stats(&self) -> ShardStorageStats {
        let cache = self.cache.lock().unwrap();

        ShardStorageStats {
            vram_count: cache.vram_cache.len(),
            vram_size_mb: cache.vram_cache_size,
            ram_count: cache.ram_cache.len(),
            ram_size_mb: cache.ram_cache_size,
            disk_count: cache.disk_cache_count,
            disk_size_mb: cache.disk_cache_size,
            network_count: cache.network_cache_count,
        }
    }

    fn store_in_disk(&self, shard_id: &str, data: &[u8]) -> AuriaResult<()> {
        let disk_path = format!("{}/{}.shard", self.disk_path, shard_id);

        // Update cache metadata
        {
            let mut cache = self.cache.lock().unwrap();
            cache.disk_cache_count += 1;
            cache.disk_cache_size += (data.len() as u32 / 1024 / 1024) as u32;
        }

        // Write to disk
        std::fs::write(disk_path, data)?;

        Ok(())
    }

    fn fetch_from_network(&self, shard_id: &str) -> AuriaResult<Vec<u8>> {
        // Simulate network fetch with random delay
        let delay = rand::random::<u64>() % 500 + 100; // 100-500ms
        std::thread::sleep(std::time::Duration::from_millis(delay));

        // Simulate network data
        let data = format!("network_data_{}", shard_id).into_bytes();
        Ok(data)
    }

    fn shard_exists_in_network(&self, shard_id: &str) -> bool {
        // Simulate network existence check
        shard_id.len() % 2 == 0 // Simple heuristic
    }
}

struct CacheManager {
    vram_cache: HashMap<String, ShardCacheEntry>,
    ram_cache: HashMap<String, ShardCacheEntry>,
    disk_cache_count: usize,
    disk_cache_size: u32,
    network_cache_count: usize,
    vram_cache_size: u32,
    ram_cache_size: u32,
}

impl CacheManager {
    fn new(config: StorageConfig) -> Self {
        Self {
            vram_cache: HashMap::new(),
            ram_cache: HashMap::new(),
            disk_cache_count: 0,
            disk_cache_size: 0,
            network_cache_count: 0,
            vram_cache_size: 0,
            ram_cache_size: 0,
        }
    }

    fn evict(&mut self, target_tier: StorageTier, target_size_mb: u32) {
        match target_tier {
            StorageTier::Vram => {
                while self.vram_cache_size > target_size_mb {
                    if let Some((shard_id, entry)) = self.vram_cache.iter().min_by_key(|(_, e)| e.last_accessed) {
                        self.vram_cache_size -= (entry.size_bytes as u32 / 1024 / 1024);
                        self.vram_cache.remove(shard_id);
                    } else {
                        break;
                    }
                }
            }
            StorageTier::Ram => {
                while self.ram_cache_size > target_size_mb {
                    if let Some((shard_id, entry)) = self.ram_cache.iter().min_by_key(|(_, e)| e.last_accessed) {
                        self.ram_cache_size -= (entry.size_bytes as u32 / 1024 / 1024);
                        self.ram_cache.remove(shard_id);
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    fn promote(&mut self, shard_id: &str, from_tier: StorageTier, to_tier: StorageTier) {
        match (from_tier, to_tier) {
            (StorageTier::Disk, StorageTier::Ram) => {
                if let Some(entry) = self.get_disk_entry(shard_id) {
                    if self.ram_cache_size + (entry.size_bytes as u32 / 1024 / 1024) <= self.ram_cache_size {
                        self.ram_cache.insert(
                            shard_id.to_string(),
                            ShardCacheEntry {
                                shard_id: shard_id.to_string(),
                                data: entry.data.clone(),
                                tier: StorageTier::Ram,
                                size_bytes: entry.size_bytes,
                                last_accessed: Instant::now(),
                                hit_count: entry.hit_count,
                            }
                        );
                        self.ram_cache_size += (entry.size_bytes as u32 / 1024 / 1024);
                    }
                }
            }
            (StorageTier::Ram, StorageTier::Vram) => {
                if let Some(entry) = self.get_ram_entry(shard_id) {
                    if self.vram_cache_size + (entry.size_bytes as u32 / 1024 / 1024) <= self.vram_cache_size {
                        self.vram_cache.insert(
                            shard_id.to_string(),
                            ShardCacheEntry {
                                shard_id: shard_id.to_string(),
                                data: entry.data.clone(),
                                tier: StorageTier::Vram,
                                size_bytes: entry.size_bytes,
                                last_accessed: Instant::now(),
                                hit_count: entry.hit_count,
                            }
                        );
                        self.vram_cache_size += (entry.size_bytes as u32 / 1024 / 1024);
                    }
                }
            }
            _ => {}
        }
    }

    fn get_disk_entry(&self, shard_id: &str) -> Option<&ShardCacheEntry> {
        // In a real implementation, this would read from disk
        None
    }

    fn get_ram_entry(&self, shard_id: &str) -> Option<&ShardCacheEntry> {
        self.ram_cache.get(shard_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_store_basic_operations() {
        let config = StorageConfig {
            vram_cache_size_mb: 100,
            ram_cache_size_mb: 500,
            disk_cache_path: "./test_cache".to_string(),
            disk_cache_size_mb: 1000,
            network_cache_enabled: false,
        };

        let store = ModelStore::new(config).unwrap();

        // Test shard storage
        let shard_id = "test_shard_123";
        let data = b"test_shard_data";

        store.store_shard(shard_id, data).unwrap();
        assert!(store.shard_exists(shard_id));

        // Test shard retrieval
        let loaded_data = store.load_shard(shard_id).unwrap();
        assert_eq!(loaded_data, data);

        // Test storage stats
        let stats = store.get_storage_stats();
        assert!(stats.vram_count > 0 || stats.ram_count > 0);
    }

    #[test]
    fn test_cache_eviction() {
        let config = StorageConfig {
            vram_cache_size_mb: 1, // Very small cache
            ram_cache_size_mb: 1,   // Very small cache
            disk_cache_path: "./test_cache".to_string(),
            disk_cache_size_mb: 1000,
            network_cache_enabled: false,
        };

        let store = ModelStore::new(config).unwrap();

        // Store multiple shards to trigger eviction
        for i in 0..10 {
            let shard_id = format!("test_shard_{}", i);
            let data = format!("data_{}", i).into_bytes();
            store.store_shard(&shard_id, &data).unwrap();
        }

        // Verify that some shards were evicted
        let stats = store.get_storage_stats();
        assert!(stats.vram_count <= 1);
        assert!(stats.ram_count <= 1);
    }

    #[test]
    fn test_tier_promotion() {
        let config = StorageConfig {
            vram_cache_size_mb: 100,
            ram_cache_size_mb: 500,
            disk_cache_path: "./test_cache".to_string(),
            disk_cache_size_mb: 1000,
            network_cache_enabled: false,
        };

        let store = ModelStore::new(config).unwrap();

        // Store shard in disk (simulated)
        let shard_id = "test_shard_promotion";
        let data = b"promotion_data";
        store.store_shard(shard_id, data).unwrap();

        // Load shard to trigger promotion to RAM
        let _ = store.load_shard(shard_id).unwrap();

        // Verify promotion happened
        let stats = store.get_storage_stats();
        assert!(stats.ram_count > 0);
    }
}