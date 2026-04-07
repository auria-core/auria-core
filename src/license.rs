use crate::shard::{PublicKey, ShardId, Signature};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub shard_id: ShardId,
    pub node_pubkey: PublicKey,
    pub expiry_timestamp: u64,
    pub signature: Signature,
}
