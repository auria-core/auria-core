//! Unit tests for AURIA Core types

use auria_core::{
    ExpertId, Hash, LicenseHash, PublicKey, RequestId, ShardId, Signature, Tensor, TensorDType,
    Tier, UsageReceipt, UsageStats,
};
use uuid::Uuid;

// Helper to create 32-byte arrays from 16-byte UUIDs
fn uuid_to_32bytes(uuid: Uuid) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&uuid.as_bytes()[..16]);
    bytes
}

#[test]
fn test_request_id_new() {
    let id = RequestId::new();
    // Should be initialized with zeros
    assert_eq!(id.0, [0u8; 16]);
}

#[test]
fn test_request_id_from_bytes() {
    let id = RequestId(*Uuid::new_v4().as_bytes());
    assert_eq!(id.0.len(), 16);
}

#[test]
fn test_request_id_clone() {
    let id = RequestId(*Uuid::new_v4().as_bytes());
    let cloned = id.clone();
    assert_eq!(id.0, cloned.0);
}

#[test]
fn test_request_id_partial_eq() {
    let id1 = RequestId(*Uuid::new_v4().as_bytes());
    let id2 = RequestId(*Uuid::new_v4().as_bytes());
    let id1_clone = RequestId(id1.0);

    assert_eq!(id1, id1_clone);
    assert_ne!(id1, id2);
}

#[test]
fn test_request_id_hash_trait() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let id1 = RequestId(*Uuid::new_v4().as_bytes());
    let id2 = RequestId(*Uuid::new_v4().as_bytes());

    set.insert(id1.clone());
    assert!(set.contains(&id1));
    assert!(!set.contains(&id2));
}

#[test]
fn test_expert_id_new() {
    let id = ExpertId::new();
    // Should have non-zero bytes from UUID
    let has_nonzero = id.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_expert_id_display() {
    let id = ExpertId(uuid_to_32bytes(Uuid::new_v4()));
    let display = format!("{}", id);
    // Should be hex encoded (64 hex chars)
    assert_eq!(display.len(), 64);
}

#[test]
fn test_expert_id_partial_eq() {
    let id1 = ExpertId(uuid_to_32bytes(Uuid::new_v4()));
    let id2 = ExpertId(uuid_to_32bytes(Uuid::new_v4()));

    assert_ne!(id1, id2);
    assert_eq!(id1, id1);
}

#[test]
fn test_expert_id_hash() {
    use std::collections::HashMap;
    let mut map = HashMap::new();

    let id = ExpertId(uuid_to_32bytes(Uuid::new_v4()));
    map.insert(id.clone(), "test_value");

    assert_eq!(map.get(&id), Some(&"test_value"));
}

#[test]
fn test_hash_new() {
    let hash = Hash::new();
    // Should have non-zero bytes
    let has_nonzero = hash.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_hash_display() {
    let hash = Hash(uuid_to_32bytes(Uuid::new_v4()));
    let display = format!("{}", hash);
    assert_eq!(display.len(), 64);
}

#[test]
fn test_hash_partial_eq() {
    let hash1 = Hash(uuid_to_32bytes(Uuid::new_v4()));
    let hash2 = Hash(uuid_to_32bytes(Uuid::new_v4()));

    assert_ne!(hash1, hash2);
    assert_eq!(hash1, hash1);
}

#[test]
fn test_hash_clone() {
    let hash = Hash(uuid_to_32bytes(Uuid::new_v4()));
    let cloned = hash.clone();
    assert_eq!(hash.0, cloned.0);
}

#[test]
fn test_signature_new() {
    let sig = Signature::new();
    // Should have non-zero bytes
    let has_nonzero = sig.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_signature_partial_eq() {
    let sig1 = Signature([1u8; 64]);
    let sig2 = Signature([2u8; 64]);
    let sig1_copy = Signature([1u8; 64]);

    assert_eq!(sig1, sig1_copy);
    assert_ne!(sig1, sig2);
}

#[test]
fn test_signature_eq() {
    let sig1 = Signature([1u8; 64]);
    let sig2 = Signature([1u8; 64]);
    assert_eq!(sig1, sig2);
}

#[test]
fn test_signature_display() {
    let sig = Signature([0xABu8; 64]);
    let display = format!("{}", sig);
    assert_eq!(display.len(), 128);
}

#[test]
fn test_signature_serialize() {
    let sig = Signature([1u8; 64]);
    let serialized = serde_json::to_string(&sig).unwrap();
    assert!(serialized.contains("AQAAAA"));
}

#[test]
fn test_signature_deserialize_valid() {
    let data = vec![1u8; 64];
    let sig: Signature = serde_json::from_slice(&data).unwrap();
    assert_eq!(sig.0, [1u8; 64]);
}

#[test]
fn test_signature_deserialize_invalid_length() {
    let data = vec![1u8; 32];
    let result: Result<Signature, _> = serde_json::from_slice(&data);
    assert!(result.is_err());
}

#[test]
fn test_signature_deserialize_empty() {
    let data = vec![];
    let result: Result<Signature, _> = serde_json::from_slice(&data);
    assert!(result.is_err());
}

#[test]
fn test_public_key_new() {
    let pk = PublicKey::new();
    // Should have non-zero bytes
    let has_nonzero = pk.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_public_key_display() {
    let pk = PublicKey(uuid_to_32bytes(Uuid::new_v4()));
    let display = format!("{}", pk);
    assert_eq!(display.len(), 64);
}

#[test]
fn test_public_key_partial_eq() {
    let pk1 = PublicKey(uuid_to_32bytes(Uuid::new_v4()));
    let pk2 = PublicKey(uuid_to_32bytes(Uuid::new_v4()));

    assert_ne!(pk1, pk2);
    assert_eq!(pk1, pk1);
}

#[test]
fn test_public_key_hash_trait() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let pk = PublicKey(uuid_to_32bytes(Uuid::new_v4()));
    set.insert(pk.clone());

    assert!(set.contains(&pk));
}

#[test]
fn test_shard_id_new() {
    let id = ShardId::new();
    // Should have non-zero bytes
    let has_nonzero = id.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_shard_id_display() {
    let id = ShardId(uuid_to_32bytes(Uuid::new_v4()));
    let display = format!("{}", id);
    assert_eq!(display.len(), 64);
}

#[test]
fn test_shard_id_partial_eq() {
    let id1 = ShardId(uuid_to_32bytes(Uuid::new_v4()));
    let id2 = ShardId(uuid_to_32bytes(Uuid::new_v4()));

    assert_ne!(id1, id2);
    assert_eq!(id1, id1);
}

#[test]
fn test_license_hash_new() {
    let lh = LicenseHash::new();
    // Should have non-zero bytes
    let has_nonzero = lh.0.iter().any(|&b| b != 0);
    assert!(has_nonzero);
}

#[test]
fn test_license_hash_display() {
    let lh = LicenseHash(uuid_to_32bytes(Uuid::new_v4()));
    let display = format!("{}", lh);
    assert_eq!(display.len(), 64);
}

#[test]
fn test_usage_stats_default() {
    let stats = UsageStats {
        tokens_generated: 100,
        tokens_processed: 50,
    };

    assert_eq!(stats.tokens_generated, 100);
    assert_eq!(stats.tokens_processed, 50);
}

#[test]
fn test_usage_stats_zero() {
    let stats = UsageStats {
        tokens_generated: 0,
        tokens_processed: 0,
    };

    assert_eq!(stats.tokens_generated, 0);
    assert_eq!(stats.tokens_processed, 0);
}

#[test]
fn test_usage_receipt_creation() {
    let request_id = RequestId(*Uuid::new_v4().as_bytes());
    let expert_ids = vec![
        ExpertId(uuid_to_32bytes(Uuid::new_v4())),
        ExpertId(uuid_to_32bytes(Uuid::new_v4())),
    ];
    let signature = Signature::new();

    let receipt = UsageReceipt {
        request_id,
        expert_ids,
        token_count: 100,
        timestamp: 1234567890,
        node_signature: signature,
    };

    assert_eq!(receipt.token_count, 100);
    assert_eq!(receipt.expert_ids.len(), 2);
}

#[test]
fn test_tensor_dtype_display() {
    assert_eq!(format!("{}", TensorDType::FP16), "FP16");
    assert_eq!(format!("{}", TensorDType::FP8), "FP8");
    assert_eq!(format!("{}", TensorDType::INT8), "INT8");
    assert_eq!(format!("{}", TensorDType::INT4), "INT4");
}

#[test]
fn test_tensor_dtype_partial_eq() {
    assert_eq!(TensorDType::FP16, TensorDType::FP16);
    assert_ne!(TensorDType::FP16, TensorDType::FP8);
    assert_ne!(TensorDType::INT8, TensorDType::INT4);
}

#[test]
fn test_tensor_creation() {
    let tensor = Tensor {
        data: vec![1u8, 2, 3, 4],
        shape: vec![2, 2],
        dtype: TensorDType::INT8,
    };

    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.dtype, TensorDType::INT8);
    assert_eq!(tensor.data.len(), 4);
}

#[test]
fn test_tensor_clone() {
    let tensor = Tensor {
        data: vec![1u8, 2, 3, 4],
        shape: vec![2, 2],
        dtype: TensorDType::FP16,
    };

    let cloned = tensor.clone();
    assert_eq!(tensor.data, cloned.data);
    assert_eq!(tensor.shape, cloned.shape);
    assert_eq!(tensor.dtype, cloned.dtype);
}

#[test]
fn test_tensor_serialize() {
    let tensor = Tensor {
        data: vec![1u8, 2],
        shape: vec![2],
        dtype: TensorDType::INT8,
    };

    let serialized = serde_json::to_string(&tensor).unwrap();
    assert!(serialized.contains("data"));
    assert!(serialized.contains("shape"));
    assert!(serialized.contains("INT8"));
}

#[test]
fn test_tensor_deserialize() {
    let json = r#"{"data":[1,2,3],"shape":[3],"dtype":"INT8"}"#;
    let tensor: Tensor = serde_json::from_str(json).unwrap();

    assert_eq!(tensor.data, vec![1, 2, 3]);
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.dtype, TensorDType::INT8);
}

// Test that Tier can be used in different contexts
#[test]
fn test_tier_as_discriminant() {
    let tier = Tier::Standard;
    let value = tier as u8;
    assert_eq!(value, 1);
}

#[test]
fn test_tier_match() {
    let tiers = vec![Tier::Nano, Tier::Standard, Tier::Pro, Tier::Max];

    for tier in tiers {
        match tier {
            Tier::Nano => assert!(true),
            Tier::Standard => assert!(true),
            Tier::Pro => assert!(true),
            Tier::Max => assert!(true),
        }
    }
}
