use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuriaError {
    #[error("Shard not found: {0:?}")]
    ShardNotFound([u8; 32]),

    #[error("Expert not found: {0:?}")]
    ExpertNotFound([u8; 32]),

    #[error("License invalid or expired for shard: {0:?}")]
    LicenseInvalid([u8; 32]),

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

    #[error("GPU error: {0}")]
    GpuError(String),
}

pub type AuriaResult<T> = std::result::Result<T, AuriaError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Tier {
    Nano = 0,
    Standard = 1,
    Pro = 2,
    Max = 3,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Nano => write!(f, "Nano"),
            Tier::Standard => write!(f, "Standard"),
            Tier::Pro => write!(f, "Pro"),
            Tier::Max => write!(f, "Max"),
        }
    }
}
