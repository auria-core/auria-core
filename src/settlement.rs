use crate::shard::{ExpertId, Signature};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub [u8; 16]);

impl RequestId {
    pub fn new() -> Self {
        Self([0u8; 16])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub tokens_generated: u64,
    pub tokens_processed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReceipt {
    pub request_id: RequestId,
    pub expert_ids: Vec<ExpertId>,
    pub token_count: u64,
    pub timestamp: u64,
    pub node_signature: Signature,
}
