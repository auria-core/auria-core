//! Tests for Expert type and related functionality

use auria_core::{Expert, ExpertId, ShardId, TensorLayout};
use uuid::Uuid;

// Helper to create 32-byte arrays from 16-byte UUIDs
fn uuid_to_32bytes(uuid: Uuid) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
    bytes
}

// Helper to create ExpertMetadata
fn make_metadata() -> auria_core::expert::ExpertMetadata {
    auria_core::expert::ExpertMetadata {
        created_at: 0,
        version: 1,
        description: None,
        tags: vec![],
        license_required: false,
    }
}

#[test]
fn test_expert_creation() {
    let expert = Expert {
        expert_id: ExpertId(uuid_to_32bytes(Uuid::new_v4())),
        shards: vec![
            ShardId(uuid_to_32bytes(Uuid::new_v4())),
            ShardId(uuid_to_32bytes(Uuid::new_v4())),
        ],
        tensor_layout: TensorLayout::new(vec![1, 2, 3]),
        metadata: make_metadata(),
    };

    assert_eq!(expert.shards.len(), 2);
}

#[test]
fn test_expert_clone() {
    let expert = Expert {
        expert_id: ExpertId(uuid_to_32bytes(Uuid::new_v4())),
        shards: vec![ShardId(uuid_to_32bytes(Uuid::new_v4()))],
        tensor_layout: TensorLayout::new(vec![]),
        metadata: make_metadata(),
    };

    let cloned = expert.clone();
    assert_eq!(expert.expert_id, cloned.expert_id);
    assert_eq!(expert.shards.len(), cloned.shards.len());
}

#[test]
fn test_expert_serialize() {
    let expert = Expert {
        expert_id: ExpertId(uuid_to_32bytes(Uuid::new_v4())),
        shards: vec![],
        tensor_layout: TensorLayout::new(vec![]),
        metadata: make_metadata(),
    };

    let serialized = serde_json::to_string(&expert).unwrap();
    assert!(serialized.contains("expert_id"));
}

#[test]
fn test_expert_with_many_shards() {
    let shard_ids: Vec<ShardId> = (0..100)
        .map(|_| ShardId(uuid_to_32bytes(Uuid::new_v4())))
        .collect();

    let expert = Expert {
        expert_id: ExpertId(uuid_to_32bytes(Uuid::new_v4())),
        shards: shard_ids,
        tensor_layout: TensorLayout::new(vec![]),
        metadata: make_metadata(),
    };

    assert_eq!(expert.shards.len(), 100);
}
