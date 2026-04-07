// File: capability_negotiation.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Cluster capability negotiation for AURIA Runtime Core.
//     Handles negotiation of hardware capabilities between nodes in a cluster,
//     ensuring optimal resource allocation and tier assignment.
//
use auria_core::{AuriaError, AuriaResult, Tier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCapability {
    pub node_id: String,
    pub hardware_profile: HardwareProfile,
    pub last_updated: u64,
    pub available_tiers: Vec<Tier>,
    pub load_factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCapabilityReport {
    pub cluster_id: String,
    pub capabilities: Vec<ClusterCapability>,
    pub timestamp: u64,
    pub cluster_tier_distribution: HashMap<Tier, u32>,
    pub average_load: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityNegotiationRequest {
    pub node_id: String,
    pub current_tier: Tier,
    pub requested_tier: Tier,
    pub reason: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityNegotiationResponse {
    pub node_id: String,
    pub approved: bool,
    pub new_tier: Tier,
    pub reason: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTierAssignment {
    pub node_id: String,
    pub assigned_tier: Tier,
    pub effective_from: u64,
    pub expires_at: u64,
    pub reason: String,
}

pub struct CapabilityNegotiator {
    cluster_id: String,
    node_id: String,
    capabilities: HashMap<String, ClusterCapability>,
    negotiation_history: Vec<CapabilityNegotiationRequest>,
    tier_assignments: Vec<ClusterTierAssignment>,
}

impl CapabilityNegotiator {
    pub fn new(cluster_id: String, node_id: String) -> Self {
        Self {
            cluster_id,
            node_id,
            capabilities: HashMap::new(),
            negotiation_history: Vec::new(),
            tier_assignments: Vec::new(),
        }
    }

    pub fn update_capability(&mut self, capability: ClusterCapability) {
        self.capabilities.insert(capability.node_id.clone(), capability);
    }

    pub fn get_cluster_report(&self) -> ClusterCapabilityReport {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let capabilities: Vec<_> = self.capabilities.values().cloned().collect();
        let cluster_tier_distribution = self.calculate_tier_distribution(&capabilities);
        let average_load = self.calculate_average_load(&capabilities);

        ClusterCapabilityReport {
            cluster_id: self.cluster_id.clone(),
            capabilities,
            timestamp,
            cluster_tier_distribution,
            average_load,
        }
    }

    pub fn request_tier_upgrade(&mut self, current_tier: Tier, requested_tier: Tier, reason: String) -> CapabilityNegotiationResponse {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let request = CapabilityNegotiationRequest {
            node_id: self.node_id.clone(),
            current_tier,
            requested_tier,
            reason,
            timestamp,
        };

        self.negotiation_history.push(request.clone());

        // Process the request
        let response = self.process_tier_request(&request);

        // If approved, update tier assignments
        if response.approved {
            self.update_tier_assignment(&response);
        }

        response
    }

    fn process_tier_request(&self, request: &CapabilityNegotiationRequest) -> CapabilityNegotiationResponse {
        // Check if the requested tier is supported by the node's hardware
        if let Some(capability) = self.capabilities.get(&request.node_id) {
            if !capability.available_tiers.contains(&request.requested_tier) {
                return CapabilityNegotiationResponse {
                    node_id: request.node_id.clone(),
                    approved: false,
                    new_tier: request.current_tier,
                    reason: format!("Requested tier {:?} not supported by hardware", request.requested_tier),
                    timestamp: request.timestamp,
                };
            }
        }

        // Check cluster load and resource availability
        let cluster_report = self.get_cluster_report();
        let load_ok = cluster_report.average_load < 0.8; // Allow upgrade if cluster load is below 80%

        if !load_ok {
            return CapabilityNegotiationResponse {
                node_id: request.node_id.clone(),
                approved: false,
                new_tier: request.current_tier,
                reason: "Cluster load too high for tier upgrade".to_string(),
                timestamp: request.timestamp,
            };
        }

        // Check if there are enough nodes at the requested tier level
        let tier_count = cluster_report.cluster_tier_distribution.get(&request.requested_tier).unwrap_or(&0);
        let tier_ok = *tier_count < 3; // Limit to 3 nodes per tier for now

        if !tier_ok {
            return CapabilityNegotiationResponse {
                node_id: request.node_id.clone(),
                approved: false,
                new_tier: request.current_tier,
                reason: format!("Too many nodes already at tier {:?}", request.requested_tier),
                timestamp: request.timestamp,
            };
        }

        // Approve the request
        CapabilityNegotiationResponse {
            node_id: request.node_id.clone(),
            approved: true,
            new_tier: request.requested_tier,
            reason: "Tier upgrade approved based on hardware capabilities and cluster load".to_string(),
            timestamp: request.timestamp,
        }
    }

    fn update_tier_assignment(&mut self, response: &CapabilityNegotiationResponse) {
        let effective_from = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expires_at = effective_from + 3600; // 1 hour expiration

        let assignment = ClusterTierAssignment {
            node_id: response.node_id.clone(),
            assigned_tier: response.new_tier,
            effective_from,
            expires_at,
            reason: response.reason.clone(),
        };

        self.tier_assignments.push(assignment);
    }

    fn calculate_tier_distribution(&self, capabilities: &[ClusterCapability]) -> HashMap<Tier, u32> {
        let mut distribution = HashMap::new();

        for capability in capabilities {
            for tier in &capability.available_tiers {
                *distribution.entry(*tier).or_insert(0) += 1;
            }
        }

        distribution
    }

    fn calculate_average_load(&self, capabilities: &[ClusterCapability]) -> f32 {
        if capabilities.is_empty() {
            return 0.0;
        }

        let total_load: f32 = capabilities.iter().map(|c| c.load_factor).sum();
        total_load / capabilities.len() as f32
    }

    pub fn get_negotiation_history(&self) -> &[CapabilityNegotiationRequest] {
        &self.negotiation_history
    }

    pub fn get_tier_assignments(&self) -> &[ClusterTierAssignment] {
        &self.tier_assignments
    }
}

pub fn negotiate_cluster_capabilities(cluster_id: String, capabilities: Vec<ClusterCapability>) -> AuriaResult<ClusterCapabilityReport> {
    let mut negotiator = CapabilityNegotiator::new(cluster_id, "cluster_manager".to_string());

    for capability in capabilities {
        negotiator.update_capability(capability);
    }

    Ok(negotiator.get_cluster_report())
}

pub fn handle_tier_request(
    cluster_id: String,
    node_id: String,
    current_tier: Tier,
    requested_tier: Tier,
    reason: String,
) -> AuriaResult<CapabilityNegotiationResponse> {
    let mut negotiator = CapabilityNegotiator::new(cluster_id, node_id);
    let response = negotiator.request_tier_upgrade(current_tier, requested_tier, reason);
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_negotiation() {
        let mut negotiator = CapabilityNegotiator::new("test-cluster".to_string(), "node-1".to_string());

        let capability1 = ClusterCapability {
            node_id: "node-1".to_string(),
            hardware_profile: HardwareProfile {
                cpu: CpuProfile {
                    vendor: "x86".to_string(),
                    brand: "Test CPU 1".to_string(),
                    cores_physical: 8,
                    cores_logical: 16,
                    frequency_mhz: 3000,
                    features: vec!["sse4.2".to_string(), "avx2".to_string()],
                },
                gpu: Some(GpuProfile {
                    name: "Test GPU 1".to_string(),
                    vendor: "NVIDIA".to_string(),
                    vram_bytes: 8 * 1024 * 1024 * 1024,
                    compute_units: 4096,
                    driver_version: "1.0".to_string(),
                    cuda_available: true,
                    metal_available: false,
                    rocm_available: false,
                }),
                ram_bytes: 32 * 1024 * 1024 * 1024,
                ram_bandwidth_gbps: 50.0,
                disk_bandwidth_mbps: 500.0,
                disk_total_bytes: 500 * 1024 * 1024 * 1024,
                network_latency_ms: 50.0,
            },
            last_updated: 1234567890,
            available_tiers: vec![Tier::Nano, Tier::Standard],
            load_factor: 0.3,
        };

        let capability2 = ClusterCapability {
            node_id: "node-2".to_string(),
            hardware_profile: HardwareProfile {
                cpu: CpuProfile {
                    vendor: "x86".to_string(),
                    brand: "Test CPU 2".to_string(),
                    cores_physical: 16,
                    cores_logical: 32,
                    frequency_mhz: 3500,
                    features: vec!["sse4.2".to_string(), "avx2".to_string(), "avx512f".to_string()],
                },
                gpu: Some(GpuProfile {
                    name: "Test GPU 2".to_string(),
                    vendor: "NVIDIA".to_string(),
                    vram_bytes: 24 * 1024 * 1024 * 1024,
                    compute_units: 6144,
                    driver_version: "2.0".to_string(),
                    cuda_available: true,
                    metal_available: false,
                    rocm_available: false,
                }),
                ram_bytes: 64 * 1024 * 1024 * 1024,
                ram_bandwidth_gbps: 80.0,
                disk_bandwidth_mbps: 1000.0,
                disk_total_bytes: 1000 * 1024 * 1024 * 1024,
                network_latency_ms: 30.0,
            },
            last_updated: 1234567890,
            available_tiers: vec![Tier::Nano, Tier::Standard, Tier::Pro],
            load_factor: 0.2,
        };

        negotiator.update_capability(capability1);
        negotiator.update_capability(capability2);

        let report = negotiator.get_cluster_report();
        assert_eq!(report.cluster_id, "test-cluster");
        assert_eq!(report.capabilities.len(), 2);
        assert_eq!(report.cluster_tier_distribution.len(), 3);
        assert!(report.average_load < 0.3);

        let response = negotiator.request_tier_upgrade(Tier::Standard, Tier::Pro, "Need more compute power".to_string());
        assert!(response.approved);
        assert_eq!(response.new_tier, Tier::Pro);
    }

    #[test]
    fn test_tier_upgrade_denial() {
        let mut negotiator = CapabilityNegotiator::new("test-cluster".to_string(), "node-1".to_string());

        let capability = ClusterCapability {
            node_id: "node-1".to_string(),
            hardware_profile: HardwareProfile {
                cpu: CpuProfile {
                    vendor: "x86".to_string(),
                    brand: "Test CPU".to_string(),
                    cores_physical: 8,
                    cores_logical: 16,
                    frequency_mhz: 3000,
                    features: vec!["sse4.2".to_string(), "avx2".to_string()],
                },
                gpu: Some(GpuProfile {
                    name: "Test GPU".to_string(),
                    vendor: "NVIDIA".to_string(),
                    vram_bytes: 8 * 1024 * 1024 * 1024,
                    compute_units: 4096,
                    driver_version: "1.0".to_string(),
                    cuda_available: true,
                    metal_available: false,
                    rocm_available: false,
                }),
                ram_bytes: 32 * 1024 * 1024 * 1024,
                ram_bandwidth_gbps: 50.0,
                disk_bandwidth_mbps: 500.0,
                disk_total_bytes: 500 * 1024 * 1024 * 1024,
                network_latency_ms: 50.0,
            },
            last_updated: 1234567890,
            available_tiers: vec![Tier::Nano, Tier::Standard],
            load_factor: 0.9, // High load
        };

        negotiator.update_capability(capability);

        let response = negotiator.request_tier_upgrade(Tier::Standard, Tier::Pro, "Need more compute power".to_string());
        assert!(!response.approved);
        assert_eq!(response.new_tier, Tier::Standard);
        assert_eq!(response.reason, "Cluster load too high for tier upgrade");
    }

    #[test]
    fn test_tier_upgrade_unsupported() {
        let mut negotiator = CapabilityNegotiator::new("test-cluster".to_string(), "node-1".to_string());

        let capability = ClusterCapability {
            node_id: "node-1".to_string(),
            hardware_profile: HardwareProfile {
                cpu: CpuProfile {
                    vendor: "x86".to_string(),
                    brand: "Test CPU".to_string(),
                    cores_physical: 8,
                    cores_logical: 16,
                    frequency_mhz: 3000,
                    features: vec!["sse4.2".to_string(), "avx2".to_string()],
                },
                gpu: Some(GpuProfile {
                    name: "Test GPU".to_string(),
                    vendor: "NVIDIA".to_string(),
                    vram_bytes: 8 * 1024 * 1024 * 1024,
                    compute_units: 4096,
                    driver_version: "1.0".to_string(),
                    cuda_available: true,
                    metal_available: false,
                    rocm_available: false,
                }),
                ram_bytes: 32 * 1024 * 1024 * 1024,
                ram_bandwidth_gbps: 50.0,
                disk_bandwidth_mbps: 500.0,
                disk_total_bytes: 500 * 1024 * 1024 * 1024,
                network_latency_ms: 50.0,
            },
            last_updated: 1234567890,
            available_tiers: vec![Tier::Nano, Tier::Standard],
            load_factor: 0.3,
        };

        negotiator.update_capability(capability);

        let response = negotiator.request_tier_upgrade(Tier::Standard, Tier::Max, "Need maximum compute power".to_string());
        assert!(!response.approved);
        assert_eq!(response.new_tier, Tier::Standard);
        assert_eq!(response.reason, "Requested tier Max not supported by hardware");
    }
}
