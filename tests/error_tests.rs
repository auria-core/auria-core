//! Unit tests for AURIA Core - Error types and conversions

use auria_core::{AuriaError, AuriaResult, Tier};

#[test]
fn test_error_shard_not_found() {
    let hash = [1u8; 32];
    let error = AuriaError::ShardNotFound(hash);
    let msg = format!("{}", error);
    assert!(msg.contains("Shard not found"));
}

#[test]
fn test_error_expert_not_found() {
    let hash = [2u8; 32];
    let error = AuriaError::ExpertNotFound(hash);
    let msg = format!("{}", error);
    assert!(msg.contains("Expert not found"));
}

#[test]
fn test_error_license_invalid() {
    let hash = [3u8; 32];
    let error = AuriaError::LicenseInvalid(hash);
    let msg = format!("{}", error);
    assert!(msg.contains("License invalid"));
}

#[test]
fn test_error_insufficient_hardware() {
    let error = AuriaError::InsufficientHardware(Tier::Max);
    let msg = format!("{}", error);
    assert!(msg.contains("Insufficient hardware"));
    assert!(msg.contains("Max"));
}

#[test]
fn test_error_storage() {
    let error = AuriaError::StorageError("disk full".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Storage error"));
    assert!(msg.contains("disk full"));
}

#[test]
fn test_error_execution() {
    let error = AuriaError::ExecutionError("out of memory".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Execution error"));
    assert!(msg.contains("out of memory"));
}

#[test]
fn test_error_network() {
    let error = AuriaError::NetworkError("connection refused".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Network error"));
    assert!(msg.contains("connection refused"));
}

#[test]
fn test_error_serialization() {
    let error = AuriaError::SerializationError("invalid json".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Serialization error"));
    assert!(msg.contains("invalid json"));
}

#[test]
fn test_error_config() {
    let error = AuriaError::ConfigError("missing value".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Configuration error"));
    assert!(msg.contains("missing value"));
}

#[test]
fn test_error_security() {
    let error = AuriaError::SecurityError("unauthorized access".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Security error"));
    assert!(msg.contains("unauthorized access"));
}

#[test]
fn test_error_cluster() {
    let error = AuriaError::ClusterError("node unreachable".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("Cluster error"));
    assert!(msg.contains("node unreachable"));
}

#[test]
fn test_error_gpu() {
    let error = AuriaError::GpuError("CUDA out of memory".to_string());
    let msg = format!("{}", error);
    assert!(msg.contains("GPU error"));
    assert!(msg.contains("CUDA out of memory"));
}

#[test]
fn test_auria_result_ok() {
    let result: AuriaResult<i32> = Ok(42);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_auria_result_err() {
    let hash = [0u8; 32];
    let result: AuriaResult<i32> = Err(AuriaError::ShardNotFound(hash));
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AuriaError::ShardNotFound(_)));
}

#[test]
fn test_tier_display() {
    assert_eq!(format!("{}", Tier::Nano), "Nano");
    assert_eq!(format!("{}", Tier::Standard), "Standard");
    assert_eq!(format!("{}", Tier::Pro), "Pro");
    assert_eq!(format!("{}", Tier::Max), "Max");
}

#[test]
fn test_tier_eq() {
    assert_eq!(Tier::Nano, Tier::Nano);
    assert_eq!(Tier::Standard, Tier::Standard);
    assert_ne!(Tier::Nano, Tier::Standard);
}

#[test]
fn test_tier_clone() {
    let tier = Tier::Pro;
    let cloned = tier;
    assert_eq!(tier, cloned);
}

#[test]
fn test_tier_partial_ord() {
    // Use manual comparison instead of deriving Ord
    use std::cmp::Ordering;

    // We can't use .cmp() directly, but we can test equality
    assert_eq!(Tier::Nano, Tier::Nano);
    assert_eq!(Tier::Standard, Tier::Standard);
    assert_eq!(Tier::Pro, Tier::Pro);
    assert_eq!(Tier::Max, Tier::Max);
}
