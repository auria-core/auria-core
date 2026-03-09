use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use uuid::Uuid;
use sha2::{Sha256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub shard_id: String,
    pub node_pubkey: String,
    pub expiry_timestamp: u64,
    pub signature: String,
    pub license_type: LicenseType,
    pub terms: LicenseTerms,
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
pub struct LicenseTerms {
    pub license_id: String,
    pub shard_id: String,
    pub node_pubkey: String,
    pub max_shards: u32,
    pub allowed_tiers: Vec<String>,
    pub rate_limit: Option<RateLimit>,
    pub expiry_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub window_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseUsage {
    pub license_id: String,
    pub node_pubkey: String,
    pub tokens_used: u64,
    pub requests_made: u64,
    pub last_updated: u64,
    pub current_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseValidationResult {
    pub valid: bool,
    pub license: Option<License>,
    pub errors: Vec<String>,
    pub node_signature: String,
}

pub struct LicenseManager {
    licenses: Arc<RwLock<HashMap<String, License>>>,
    usage: Arc<RwLock<HashMap<String, LicenseUsage>>>,
    rate_limits: Arc<RwLock<HashMap<String, Vec<Instant>>>,
}

impl LicenseManager {
    pub fn new() -> Self {
        Self {
            licenses: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_license(&self, license: License) -> AuriaResult<()�3e {
        // Validate license signature
        if !self.verify_license_signature(&license) {
            return Err(AuriaError::LicenseInvalid("Invalid license signature".to_string()));
        }

        // Check license expiry
        if license.expiry_timestamp < Self::current_timestamp() {
            return Err(AuriaError::LicenseInvalid("License has expired".to_string()));
        }

        // Store license
        let mut licenses = self.licenses.write().unwrap();
        licenses.insert(license.shard_id.clone(), license);

        Ok(())
    }

    pub fn validate_license_for_shard(&self, shard_id: &str, node_pubkey: &str) -> AuriaResult<LicenseValidationResult> {
        let licenses = self.licenses.read().unwrap();
        let license = licenses.get(shard_id);

        // Check if license exists
        if license.is_none() {
            return Ok(LicenseValidationResult {
                valid: false,
                license: None,
                errors: vec![format!("No license found for shard {}", shard_id)],
                node_signature: Self::generate_node_signature(node_pubkey),
            });
        }

        let license = license.unwrap().clone();

        // Check node authorization
        if license.node_pubkey != node_pubkey {
            return Ok(LicenseValidationResult {
                valid: false,
                license: Some(license),
                errors: vec![format!("Node {} not authorized for shard {}", node_pubkey, shard_id)],
                node_signature: Self::generate_node_signature(node_pubkey),
            });
        }

        // Check rate limits
        if let Some(rate_limit) = license.terms.rate_limit.as_ref() {
            if !self.check_rate_limit(shard_id, rate_limit) {
                return Ok(LicenseValidationResult {
                    valid: false,
                    license: Some(license),
                    errors: vec![format!("Rate limit exceeded for shard {}", shard_id)],
                    node_signature: Self::generate_node_signature(node_pubkey),
                });
            }
        }

        // Check usage limits
        if !self.check_usage_limits(&license) {
            return Ok(LicenseValidationResult {
                valid: false,
                license: Some(license),
                errors: vec![format!("Usage limits exceeded for shard {}", shard_id)],
                node_signature: Self::generate_node_signature(node_pubkey),
            });
        }

        // License is valid
        Ok(LicenseValidationResult {
            valid: true,
            license: Some(license),
            errors: vec![],
            node_signature: Self::generate_node_signature(node_pubkey),
        })
    }

    pub fn validate_license(&self, license: &License) -> bool {
        self.verify_license_signature(license) &&
        license.expiry_timestamp > Self::current_timestamp() &&
        self.check_usage_limits(license)
    }

    pub fn license_valid_for_shard(&self, shard_id: &str) -> bool {
        let licenses = self.licenses.read().unwrap();
        if let Some(license) = licenses.get(shard_id) {
            return self.validate_license(license);
        }
        false
    }

    pub fn record_usage(&self, shard_id: &str, node_pubkey: &str, tokens: u32) -> AuriaResult<()�3e {
        let mut usage = self.usage.write().unwrap();
        let key = format!("{}-{}", shard_id, node_pubkey);

        let mut license_usage = usage.entry(key.clone()).or_insert(LicenseUsage {
            license_id: shard_id.to_string(),
            node_pubkey: node_pubkey.to_string(),
            tokens_used: 0,
            requests_made: 0,
            last_updated: Self::current_timestamp(),
            current_rate: 0,
        });

        license_usage.tokens_used += tokens as u64;
        license_usage.requests_made += 1;
        license_usage.last_updated = Self::current_timestamp();

        // Update rate limits
        self.update_rate_limits(shard_id, tokens);

        Ok(())
    }

    fn verify_license_signature(&self, license: &License) -> bool {
        // Simulate signature verification
        // In a real implementation, this would use cryptographic verification
        let expected_signature = Self::generate_license_signature(license);
        expected_signature == license.signature
    }

    fn check_rate_limit(&self, shard_id: &str, rate_limit: &RateLimit) -> bool {
        let mut rate_limits = self.rate_limits.write().unwrap();
        let now = Instant::now();
        let key = format!("{}-{}", shard_id, rate_limit.requests_per_second);

        // Clean up old timestamps
        rate_limits.entry(key.clone()).or_insert(Vec::new()).retain(|t| now.duration_since(*t).as_secs() < rate_limit.window_seconds as u64);

        let timestamps = rate_limits.entry(key).or_insert(Vec::new());

        if timestamps.len() >= rate_limit.requests_per_second as usize {
            return false;
        }

        timestamps.push(now);
        true
    }

    fn check_usage_limits(&self, license: &License) -> bool {
        let usage = self.usage.read().unwrap();
        let key = format!("{}-{}", license.shard_id, license.node_pubkey);

        if let Some(license_usage) = usage.get(&key) {
            match &license.license_type {
                LicenseType::Subscription { max_requests_per_day, .. } => {
                    let requests_today = license_usage.requests_made;
                    requests_today <= *max_requests_per_day as u64
                }
                LicenseType::PayPerUse { credits, .. } => {
                    let credits_used = (license_usage.tokens_used as f64 / 100.0) as u64; // 100 tokens per credit
                    credits_used <= *credits as u64
                }
                LicenseType::Enterprise { max_concurrent_requests, .. } => {
                    license_usage.requests_made <= *max_concurrent_requests as u64
                }
                LicenseType::Community { .. } => true,
            }
        } else {
            true
        }
    }

    fn update_rate_limits(&self, shard_id: &str, tokens: u32) {
        let licenses = self.licenses.read().unwrap();
        if let Some(license) = licenses.get(shard_id) {
            if let Some(rate_limit) = license.terms.rate_limit.as_ref() {
                let mut rate_limits = self.rate_limits.write().unwrap();
                let key = format!("{}-{}", shard_id, rate_limit.requests_per_second);
                rate_limits.entry(key).or_insert(Vec::new()).push(Instant::now());
            }
        }
    }

    fn generate_license_signature(license: &License) -> String {
        // Simple hash-based signature for demonstration
        let mut hasher = Sha256::new();
        hasher.update(license.shard_id.as_bytes());
        hasher.update(license.node_pubkey.as_bytes());
        hasher.update(license.expiry_timestamp.to_le_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    fn generate_node_signature(node_pubkey: &str) -> String {
        // Simple hash-based signature for demonstration
        let mut hasher = Sha256::new();
        hasher.update(node_pubkey.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_manager_basic_operations() {
        let manager = LicenseManager::new();

        // Create a test license
        let license = License {
            shard_id: "test_shard_123".to_string(),
            node_pubkey: "test_node_456".to_string(),
            expiry_timestamp: Self::current_timestamp() + 86400 * 365, // 1 year from now
            signature: "test_signature".to_string(),
            license_type: LicenseType::PayPerUse {
                credits: 1000,
                cost_per_token: 0.01,
            },
            terms: LicenseTerms {
                license_id: "license_123".to_string(),
                shard_id: "test_shard_123".to_string(),
                node_pubkey: "test_node_456".to_string(),
                max_shards: 1,
                allowed_tiers: vec!["nano".to_string(), "standard".to_string()],
                rate_limit: Some(RateLimit {
                    requests_per_second: 10,
                    burst_size: 20,
                    window_seconds: 60,
                }),
                expiry_timestamp: Self::current_timestamp() + 86400 * 365,
            },
        };

        // Add license
        manager.add_license(license.clone()).unwrap();

        // Validate license
        let result = manager.validate_license_for_shard("test_shard_123", "test_node_456").unwrap();
        assert!(result.valid);
        assert_eq!(result.errors.len(), 0);

        // Record usage
        manager.record_usage("test_shard_123", "test_node_456", 100).unwrap();

        // Check usage
        let licenses = manager.licenses.read().unwrap();
        assert!(licenses.contains_key("test_shard_123"));
    }

    #[test]
    fn test_license_validation_failures() {
        let manager = LicenseManager::new();

        // Test invalid signature
        let invalid_license = License {
            shard_id: "test_shard_123".to_string(),
            node_pubkey: "test_node_456".to_string(),
            expiry_timestamp: Self::current_timestamp() + 86400,
            signature: "invalid_signature".to_string(),
            license_type: LicenseType::Community {
                tier: "standard".to_string(),
            },
            terms: LicenseTerms {
                license_id: "license_123".to_string(),
                shard_id: "test_shard_123".to_string(),
                node_pubkey: "test_node_456".to_string(),
                max_shards: 1,
                allowed_tiers: vec!["nano".to_string(), "standard".to_string()],
                rate_limit: None,
                expiry_timestamp: Self::current_timestamp() + 86400,
            },
        };

        let result = manager.validate_license_for_shard("test_shard_123", "test_node_456").unwrap();
        assert!(!result.valid);
        assert!(result.errors.contains(&"Invalid license signature".to_string()));

        // Test expired license
        let expired_license = License {
            expiry_timestamp: Self::current_timestamp() - 100, // Expired
            ..invalid_license.clone()
        };

        manager.add_license(expired_license).unwrap();
        let result = manager.validate_license_for_shard("test_shard_123", "test_node_456").unwrap();
        assert!(!result.valid);
        assert!(result.errors.contains(&"License has expired".to_string()));
    }

    #[test]
    fn test_rate_limiting() {
        let manager = LicenseManager::new();

        let license = License {
            shard_id: "test_shard_rate_limit".to_string(),
            node_pubkey: "test_node_rate_limit".to_string(),
            expiry_timestamp: Self::current_timestamp() + 86400,
            signature: "test_signature".to_string(),
            license_type: LicenseType::Subscription {
                tier: "standard".to_string(),
                max_requests_per_day: 1000,
            },
            terms: LicenseTerms {
                license_id: "license_rate_limit".to_string(),
                shard_id: "test_shard_rate_limit".to_string(),
                node_pubkey: "test_node_rate_limit".to_string(),
                max_shards: 1,
                allowed_tiers: vec!["standard".to_string()],
                rate_limit: Some(RateLimit {
                    requests_per_second: 5,
                    burst_size: 10,
                    window_seconds: 60,
                }),
                expiry_timestamp: Self::current_timestamp() + 86400,
            },
        };

        manager.add_license(license).unwrap();

        // Make 5 requests quickly (should succeed)
        for _ in 0..5 {
            let result = manager.validate_license_for_shard("test_shard_rate_limit", "test_node_rate_limit").unwrap();
            assert!(result.valid);
        }

        // Make 6th request (should fail due to rate limit)
        let result = manager.validate_license_for_shard("test_shard_rate_limit", "test_node_rate_limit").unwrap();
        assert!(!result.valid);
        assert!(result.errors.contains(&"Rate limit exceeded for shard test_shard_rate_limit".to_string()));
    }
}

pub use self::License;
pub use self::LicenseType;
pub use self::LicenseTerms;
pub use self::RateLimit;
pub use self::LicenseUsage;
pub use self::LicenseValidationResult;
pub use self::LicenseManager;