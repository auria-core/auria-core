use serde::{Deserialize, Serialize, de::Error};
use std::fmt;
use uuid::Uuid;
use chrono::Utc;
use std::collections::HashMap;
use crate::tensor::{Tensor, TensorDType};

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
    pub tags: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub [u8; 32]);

impl ShardId {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertId(pub [u8; 32]);

impl ExpertId {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for ExpertId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKey(pub [u8; 32]);

impl PublicKey {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone)]
pub struct Signature(pub [u8; 64]);

impl serde::Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<u8>::deserialize(deserializer)?;
        if vec.len() != 64 {
            return Err(D::Error::custom(format!("signature must be 64 bytes, got {}", vec.len())));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&vec);
        Ok(Signature(arr))
    }
}

impl Signature {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 64];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardDefinition {
    pub shard_id: ShardId,
    pub expert_id: ExpertId,
    pub tensor_shape: Vec<u32>,
    pub tensor_dtype: TensorDType,
    pub required_license: Option<LicenseHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseHash(pub [u8; 32]);

impl LicenseHash {
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
        Self(bytes)
    }
}

impl fmt::Display for LicenseHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardIndex {
    pub shards: HashMap<ShardId, ShardDefinition>,
    pub experts: HashMap<ExpertId, Vec<ShardId>>,
}

impl ShardIndex {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            experts: HashMap::new(),
        }
    }

    pub fn add_shard(&mut self, shard: ShardDefinition) {
        self.shards.insert(shard.shard_id.clone(), shard.clone());
        self.experts.entry(shard.expert_id.clone())
            .or_insert_with(Vec::new)
            .push(shard.shard_id.clone());
    }

    pub fn get_shard(&self, shard_id: &ShardId) -> Option<&ShardDefinition> {
        self.shards.get(shard_id)
    }

    pub fn get_expert(&self, expert_id: &ExpertId) -> Option<&Vec<ShardId>> {
        self.experts.get(expert_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub hash: Hash,
}

pub type ShardResult<T> = std::result::Result<T, ShardError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardError {
    InvalidHash,
    MissingMetadata,
    InvalidTensor,
    InvalidShape,
    InvalidDtype,
    OwnerMismatch,
    LicenseInvalid,
    StorageError(String),
}