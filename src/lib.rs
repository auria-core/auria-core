// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Core types and data structures for the AURIA Runtime Core.
//     This crate defines the fundamental types used across all AURIA components
//     including Tensor, Shard, Expert, License, HardwareProfile, and error types.
//
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

macro_rules! impl_hex_serialize {
    ($ty:ty, $len:expr) => {
        impl Serialize for $ty {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&hex::encode(self.0))
            }
        }
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let s = String::deserialize(deserializer)?;
                let bytes = hex::decode(s).map_err(|e| serde::de::Error::custom(e))?;
                if bytes.len() != $len {
                    return Err(serde::de::Error::custom("invalid length"));
                }
                let mut arr = [0u8; $len];
                arr.copy_from_slice(&bytes);
                Ok(Self(arr))
            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    pub dtype: TensorDType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDType {
    FP16,
    FP8,
    INT8,
    INT4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub shard_id: ShardId,
    pub expert_id: ExpertId,
    pub tensor: Tensor,
    pub metadata: ShardMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub owner: PublicKey,
    pub license_hash: Option<Hash>,
    pub created_at: u64,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub expert_id: ExpertId,
    pub shards: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLayout {
    pub offset: u64,
    pub stride: u32,
    pub shape: Vec<u32>,
}

impl_hex_serialize!(ShardId, 32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardId(pub [u8; 32]);

impl_hex_serialize!(ExpertId, 32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertId(pub [u8; 32]);

impl_hex_serialize!(PublicKey, 32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PublicKey(pub [u8; 32]);

impl_hex_serialize!(Signature, 64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signature(pub [u8; 64]);

impl_hex_serialize!(Hash, 32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash(pub [u8; 32]);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RuntimeVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl RuntimeVersion {
    pub fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Nano,
    Standard,
    Pro,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuProfile,
    pub gpu: Option<GpuProfile>,
    pub ram_bytes: u64,
    pub ram_bandwidth_gbps: f32,
    pub disk_bandwidth_mbps: f32,
    pub network_latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub vendor: String,
    pub brand: String,
    pub cores: u32,
    pub threads: u32,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub vendor: String,
    pub name: String,
    pub vram_bytes: u64,
    pub compute_capability: (u8, u8),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub shard_id: ShardId,
    pub node_pubkey: PublicKey,
    pub expiry_timestamp: u64,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDefinition {
    pub expert_id: ExpertId,
    pub shard_ids: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReceipt {
    pub request_id: RequestId,
    pub expert_ids: Vec<ExpertId>,
    pub token_count: u32,
    pub timestamp: u64,
    pub node_signature: Signature,
}

impl_hex_serialize!(RequestId, 16);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub [u8; 16]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub expert_ids: Vec<ExpertId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutput {
    pub tokens: Vec<String>,
    pub usage: UsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub tokens_generated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub position: u32,
    pub kv_cache: Vec<Tensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDefinition {
    pub expert_id: ExpertId,
    pub shard_ids: Vec<ShardId>,
    pub tensor_layout: TensorLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReceipt {
    pub request_id: RequestId,
    pub expert_ids: Vec<ExpertId>,
    pub token_count: u32,
    pub timestamp: u64,
    pub node_signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuProfile,
    pub gpu: Option<GpuProfile>,
    pub ram_bytes: u64,
    pub ram_bandwidth_gbps: f32,
    pub disk_bandwidth_mbps: f32,
    pub network_latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub vendor: String,
    pub brand: String,
    pub cores: u32,
    pub threads: u32,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub vendor: String,
    pub name: String,
    pub vram_bytes: u64,
    pub compute_capability: (u8, u8),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseTerms {
    pub license_type: LicenseType,
    pub max_shards: u32,
    pub allowed_tiers: Vec<String>,
    pub rate_limit: Option<RateLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LicenseType {
    Subscription {
        tier: String,
        max_requests_per_day: u64,
    },
    PayPerUse {
        credits: u64,
        cost_per_token: f64,
    },
    Enterprise {
        unlimited: bool,
        max_concurrent_requests: u32,
    },
    Community {
        tier: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseUsage {
    pub license_id: ShardId,
    pub node_pubkey: PublicKey,
    pub tokens_used: u64,
    pub requests_made: u64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageTier {
    Vram,
    Ram,
    Disk,
    Network,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTierConfig {
    pub tier: StorageTier,
    pub max_size_bytes: u64,
    pub path: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub vram_count: usize,
    pub ram_count: usize,
    pub disk_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertCacheEntry {
    pub expert_id: ExpertId,
    pub tensor: Tensor,
    pub last_used_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLayout {
    pub offset: u64,
    pub stride: u32,
    pub shape: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub owner: PublicKey,
    pub license_hash: Option<Hash>,
    pub created_at: u64,
    pub version: u32,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Nano => write!(f, "Nano"),
            Tier::Standard => write!(f, "Standard"),
            Tier::Pro => write!(f, "Pro"),
            Tier::Max => write!(f, "Max"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuriaError {
    #[error("Shard not found: {0:?}")]
    ShardNotFound(ShardId),

    #[error("Expert not found: {0:?}")]
    ExpertNotFound(ExpertId),

    #[error("License invalid or expired for shard: {0:?}")]
    LicenseInvalid(ShardId),

    #[error("Insufficient hardware capabilities for tier: {0}")]
    InsufficientHardware(Tier),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Security error: {0}")]
    SecurityError(String),

    #[error("Cluster error: {0}")]
    ClusterError(String),
}

pub type AuriaResult<T> = std::result::Result<T, AuriaError>;
