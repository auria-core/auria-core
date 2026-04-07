// File: hardware_profiler.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Hardware profiling for AURIA Runtime Core.
//     Detects and quantifies node hardware capabilities including CPU, GPU,
//     RAM, disk, and network. Used to determine supported execution tiers.
//
use auria_core::{AuriaError, AuriaResult, Tier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub vendor: String,
    pub brand: String,
    pub cores_physical: u32,
    pub cores_logical: u32,
    pub frequency_mhz: u64,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub name: String,
    pub vendor: String,
    pub vram_bytes: u64,
    pub compute_units: u32,
    pub driver_version: String,
    pub cuda_available: bool,
    pub metal_available: bool,
    pub rocm_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuProfile,
    pub gpu: Option<GpuProfile>,
    pub ram_bytes: u64,
    pub ram_bandwidth_gbps: f32,
    pub disk_bandwidth_mbps: f32,
    pub disk_total_bytes: u64,
    pub network_latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfiguration {
    pub enabled_tiers: Vec<Tier>,
    pub recommended_batch_size: u32,
    pub max_concurrent_requests: u32,
}

pub struct HardwareProfiler {
    profile: HardwareProfile,
}

impl HardwareProfiler {
    pub fn new() -> AuriaResult<Self> {
        let profile = detect_hardware()?;
        Ok(Self { profile })
    }

    pub fn get_profile(&self) -> &HardwareProfile {
        &self.profile
    }

    pub fn get_tier_configuration(&self) -> TierConfiguration {
        determine_tiers(&self.profile)
    }

    pub fn is_tier_supported(&self, tier: Tier) -> bool {
        self.get_tier_configuration().enabled_tiers.contains(&tier)
    }
}

pub fn detect_hardware() -> AuriaResult<HardwareProfile> {
    let start_time = Instant::now();

    let cpu = detect_cpu()?;
    let gpu = detect_gpu();
    let ram_bytes = detect_ram()?;
    let (disk_bandwidth_mbps, disk_total_bytes) = detect_disk()?;
    let network_latency_ms = measure_network_latency()?;

    let ram_bandwidth_gbps = estimate_ram_bandwidth(&cpu);

    log::info!("Hardware detection completed in {:.2?} seconds", start_time.elapsed());

    Ok(HardwareProfile {
        cpu,
        gpu,
        ram_bytes,
        ram_bandwidth_gbps,
        disk_bandwidth_mbps,
        disk_total_bytes,
        network_latency_ms,
    })
}

fn detect_cpu() -> AuriaResult<CpuProfile> {
    let vendor = std::env::consts::ARCH.to_string();
    let brand = detect_cpu_brand();
    let cores_physical = num_cpus::get_physical() as u32;
    let cores_logical = num_cpus::get() as u32;
    let frequency_mhz = detect_cpu_frequency();
    let features = detect_cpu_features();

    Ok(CpuProfile {
        vendor,
        brand,
        cores_physical,
        cores_logical,
        frequency_mhz,
        features,
    })
}

fn detect_cpu_brand() -> String {
    #[cfg(target_os = "windows")]
    {
        detect_cpu_brand_windows()
    }
    #[cfg(target_os = "linux")]
    {
        detect_cpu_brand_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_cpu_brand_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        std::env::consts::ARCH.to_string()
    }
}

#[cfg(target_os = "windows")]
fn detect_cpu_brand_windows() -> String {
    if let Ok(output) = Command::new("wmic")
        .args(&["cpu", "get", "name"])
        .output()
    {
        if let Ok(brand) = String::from_utf8(output.stdout) {
            return brand.lines().nth(1).unwrap_or("Unknown CPU").to_string();
        }
    }
    "Unknown CPU".to_string()
}

#[cfg(target_os = "linux")]
fn detect_cpu_brand_linux() -> String {
    if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.starts_with("model name") {
                return line.split(':').nth(1).unwrap_or("Unknown CPU").trim().to_string();
            }
        }
    }
    "Unknown CPU".to_string()
}

#[cfg(target_os = "macos")]
fn detect_cpu_brand_macos() -> String {
    if let Ok(output) = Command::new("sysctl")
        .args(&["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if let Ok(brand) = String::from_utf8(output.stdout) {
            return brand.trim().to_string();
        }
    }
    "Unknown CPU".to_string()
}

fn detect_cpu_frequency() -> u64 {
    #[cfg(target_os = "windows")]
    {
        detect_cpu_frequency_windows()
    }
    #[cfg(target_os = "linux")]
    {
        detect_cpu_frequency_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_cpu_frequency_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        3000 // Default frequency in MHz
    }
}

#[cfg(target_os = "windows")]
fn detect_cpu_frequency_windows() -> u64 {
    if let Ok(output) = Command::new("wmic")
        .args(&["cpu", "get", "maxclockspeed"])
        .output()
    {
        if let Ok(freq) = String::from_utf8(output.stdout) {
            return freq
                .lines()
                .nth(1)
                .and_then(|line| line.trim().parse::<u64>().ok())
                .unwrap_or(3000);
        }
    }
    3000
}

#[cfg(target_os = "linux")]
fn detect_cpu_frequency_linux() -> u64 {
    if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.starts_with("cpu MHz") {
                return line
                    .split(':')
                    .nth(1)
                    .and_then(|val| val.trim().parse::<u64>().ok())
                    .unwrap_or(3000);
            }
        }
    }
    3000
}

#[cfg(target_os = "macos")]
fn detect_cpu_frequency_macos() -> u64 {
    if let Ok(output) = Command::new("sysctl")
        .args(&["-n", "hw.cpufrequency"])
        .output()
    {
        if let Ok(freq) = String::from_utf8(output.stdout) {
            return freq
                .trim()
                .parse::<u64>()
                .map(|freq| freq / 1_000_000) // Convert Hz to MHz
                .unwrap_or(3000);
        }
    }
    3000
}

fn detect_cpu_features() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        detect_cpu_features_x86()
    }
    #[cfg(target_arch = "aarch64")]
    {
        detect_cpu_features_arm()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vec!["generic".to_string()]
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_cpu_features_x86() -> Vec<String> {
    let mut features = Vec::new();

    // Check for SSE4.2
    if is_x86_feature_detected!("sse4.2") {
        features.push("sse4.2".to_string());
    }

    // Check for AVX2
    if is_x86_feature_detected!("avx2") {
        features.push("avx2".to_string());
    }

    // Check for AVX512
    if is_x86_feature_detected!("avx512f") {
        features.push("avx512f".to_string());
    }

    features
}

#[cfg(target_arch = "aarch64")]
fn detect_cpu_features_arm() -> Vec<String> {
    let mut features = Vec::new();

    // Check for NEON
    if is_aarch64_feature_detected!("neon") {
        features.push("neon".to_string());
    }

    // Check for SVE
    if is_aarch64_feature_detected!("sve") {
        features.push("sve".to_string());
    }

    features
}

fn detect_gpu() -> Option<GpuProfile> {
    #[cfg(target_os = "windows")]
    {
        detect_gpu_windows()
    }
    #[cfg(target_os = "linux")]
    {
        detect_gpu_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_gpu_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
fn detect_gpu_windows() -> Option<GpuProfile> {
    // Use DirectX to detect GPUs
    if let Ok(output) = Command::new("dxdiag")
        .args(&["/t", "dxdiag_output.txt"])
        .output()
    {
        if let Ok(content) = fs::read_to_string("dxdiag_output.txt") {
            let mut gpu = None;
            for line in content.lines() {
                if line.contains("Card name:") {
                    let name = line.split(':').nth(1).unwrap_or("Unknown").trim().to_string();
                    gpu = Some(GpuProfile {
                        name,
                        vendor: detect_gpu_vendor(&name),
                        vram_bytes: 4 * 1024 * 1024 * 1024, // Default 4GB
                        compute_units: 2048,
                        driver_version: "Unknown".to_string(),
                        cuda_available: false,
                        metal_available: false,
                        rocm_available: false,
                    });
                    break;
                }
            }
            fs::remove_file("dxdiag_output.txt").ok();
            return gpu;
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_gpu_linux() -> Option<GpuProfile> {
    // Check for NVIDIA GPUs
    if Path::new("/proc/driver/nvidia").exists() {
        return detect_nvidia_gpu_linux();
    }

    // Check for AMD GPUs
    if Path::new("/sys/class/drm").exists() {
        return detect_amd_gpu_linux();
    }

    None
}

#[cfg(target_os = "linux")]
fn detect_nvidia_gpu_linux() -> Option<GpuProfile> {
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_capability,driver_version")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if let Ok(info) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = info.split(',').collect();
            if parts.len() >= 4 {
                let name = parts[0].trim().to_string();
                let vram_mb = parts[1].trim().parse::<u64>().unwrap_or(4096);
                let compute_capability = parts[2].trim();
                let driver_version = parts[3].trim().to_string();

                let compute_units = match compute_capability {
                    "7.0" => 3584,
                    "7.5" => 4608,
                    "8.0" => 8192,
                    _ => 2048,
                };

                return Some(GpuProfile {
                    name,
                    vendor: "NVIDIA".to_string(),
                    vram_bytes: vram_mb * 1024 * 1024,
                    compute_units,
                    driver_version,
                    cuda_available: true,
                    metal_available: false,
                    rocm_available: false,
                });
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_amd_gpu_linux() -> Option<GpuProfile> {
    if let Ok(output) = Command::new("rocm-smi")
        .arg("--showproductname")
        .output()
    {
        if let Ok(name) = String::from_utf8(output.stdout) {
            return Some(GpuProfile {
                name: name.trim().to_string(),
                vendor: "AMD".to_string(),
                vram_bytes: 8 * 1024 * 1024 * 1024, // Default 8GB
                compute_units: 2560,
                driver_version: "Unknown".to_string(),
                cuda_available: false,
                metal_available: false,
                rocm_available: true,
            });
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_gpu_macos() -> Option<GpuProfile> {
    if let Ok(output) = Command::new("system_profiler")
        .args(&["SPDisplaysDataType"])
        .output()
    {
        if let Ok(content) = String::from_utf8(output.stdout) {
            for line in content.lines() {
                if line.contains("Chipset Model:") {
                    let name = line.split(':').nth(1).unwrap_or("Unknown").trim().to_string();
                    return Some(GpuProfile {
                        name,
                        vendor: "Apple".to_string(),
                        vram_bytes: 8 * 1024 * 1024 * 1024, // Default 8GB
                        compute_units: 2048,
                        driver_version: "Metal".to_string(),
                        cuda_available: false,
                        metal_available: true,
                        rocm_available: false,
                    });
                }
            }
        }
    }
    None
}

fn detect_gpu_vendor(gpu_name: &str) -> String {
    if gpu_name.to_lowercase().contains("nvidia") {
        "NVIDIA".to_string()
    } else if gpu_name.to_lowercase().contains("amd") || gpu_name.to_lowercase().contains("radeon") {
        "AMD".to_string()
    } else if gpu_name.to_lowercase().contains("apple") {
        "Apple".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn detect_ram() -> AuriaResult<u64> {
    #[cfg(target_os = "windows")]
    {
        detect_ram_windows()
    }
    #[cfg(target_os = "linux")]
    {
        detect_ram_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_ram_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        Ok(16 * 1024 * 1024 * 1024) // Default 16GB
    }
}

#[cfg(target_os = "windows")]
fn detect_ram_windows() -> AuriaResult<u64> {
    if let Ok(output) = Command::new("wmic")
        .args(&["computersystem", "get", "totalphysicalmemory"])
        .output()
    {
        if let Ok(memory) = String::from_utf8(output.stdout) {
            return memory
                .lines()
                .nth(1)
                .and_then(|line| line.trim().parse::<u64>().ok())
                .map(Ok)
                .unwrap_or_else(|| Err(AuriaError::ConfigError("Failed to parse RAM size".to_string())));
        }
    }
    Ok(16 * 1024 * 1024 * 1024) // Default 16GB
}

#[cfg(target_os = "linux")]
fn detect_ram_linux() -> AuriaResult<u64> {
    if let Ok(content) = fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1]
                        .parse::<u64>()
                        .map(|kb| kb * 1024) // Convert KB to bytes
                        .map_err(|_| AuriaError::ConfigError("Failed to parse RAM size".to_string()));
                }
            }
        }
    }
    Ok(16 * 1024 * 1024 * 1024) // Default 16GB
}

#[cfg(target_os = "macos")]
fn detect_ram_macos() -> AuriaResult<u64> {
    if let Ok(output) = Command::new("sysctl")
        .args(&["-n", "hw.memsize"])
        .output()
    {
        if let Ok(memory) = String::from_utf8(output.stdout) {
            return memory
                .trim()
                .parse::<u64>()
                .map_err(|_| AuriaError::ConfigError("Failed to parse RAM size".to_string()));
        }
    }
    Ok(16 * 1024 * 1024 * 1024) // Default 16GB
}

fn detect_disk() -> AuriaResult<(f32, u64)> {
    #[cfg(target_os = "windows")]
    {
        detect_disk_windows()
    }
    #[cfg(target_os = "linux")]
    {
        detect_disk_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_disk_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        Ok((500.0, 500 * 1024 * 1024 * 1024)) // Default 500MB/s, 500GB
    }
}

#[cfg(target_os = "windows")]
fn detect_disk_windows() -> AuriaResult<(f32, u64)> {
    // Use PowerShell to get disk information
    if let Ok(output) = Command::new("powershell")
        .args(&["Get-WmiObject", "Win32_LogicalDisk", "|", "Select-Object", "Size,FreeSpace"])
        .output()
    {
        if let Ok(content) = String::from_utf8(output.stdout) {
            // Parse the output to get disk size and calculate bandwidth
            // For now, return default values
            return Ok((500.0, 500 * 1024 * 1024 * 1024));
        }
    }
    Ok((500.0, 500 * 1024 * 1024 * 1024)) // Default 500MB/s, 500GB
}

#[cfg(target_os = "linux")]
fn detect_disk_linux() -> AuriaResult<(f32, u64)> {
    if let Ok(content) = fs::read_to_string("/proc/partitions") {
        let mut total_size = 0;
        for line in content.lines().skip(2) { // Skip header lines
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(size_kb) = parts[2].parse::<u64>() {
                    total_size += size_kb * 1024; // Convert KB to bytes
                }
            }
        }
        return Ok((500.0, total_size)); // Default 500MB/s
    }
    Ok((500.0, 500 * 1024 * 1024 * 1024)) // Default 500MB/s, 500GB
}

#[cfg(target_os = "macos")]
fn detect_disk_macos() -> AuriaResult<(f32, u64)> {
    if let Ok(output) = Command::new("diskutil")
        .arg("list")
        .output()
    {
        if let Ok(content) = String::from_utf8(output.stdout) {
            // Parse diskutil output to get disk sizes
            // For now, return default values
            return Ok((500.0, 500 * 1024 * 1024 * 1024));
        }
    }
    Ok((500.0, 500 * 1024 * 1024 * 1024)) // Default 500MB/s, 500GB
}

fn measure_network_latency() -> AuriaResult<f32> {
    // Try multiple endpoints for redundancy
    let endpoints = [
        "https://www.google.com",
        "https://www.cloudflare.com",
        "https://www.amazon.com",
    ];

    for endpoint in &endpoints {
        if let Ok(latency) = measure_latency_to_endpoint(endpoint) {
            return Ok(latency);
        }
    }

    // Fallback to default value
    Ok(50.0)
}

fn measure_latency_to_endpoint(endpoint: &str) -> AuriaResult<f32> {
    let start_time = Instant::now();

    let response = reqwest::blocking::get(endpoint);

    let elapsed = start_time.elapsed();
    let latency_ms = elapsed.as_secs_f32() * 1000.0;

    if response.is_ok() {
        Ok(latency_ms)
    } else {
        Err(AuriaError::NetworkError(format!("Failed to measure latency to {}: {}", endpoint, response.err().unwrap_or("Unknown error"))))
    }
}

fn estimate_ram_bandwidth(cpu: &CpuProfile) -> f32 {
    // Simple estimation based on CPU cores
    let gb_per_core = 5.0;
    (cpu.cores_logical as f32) * gb_per_core
}

fn determine_tiers(profile: &HardwareProfile) -> TierConfiguration {
    let mut enabled_tiers = Vec::new();

    // Nano tier: minimum requirements
    if profile.ram_bytes >= 8 * 1024 * 1024 * 1024 {
        enabled_tiers.push(Tier::Nano);
    }

    // Standard tier: GPU with 8GB VRAM or 32GB RAM
    if let Some(ref gpu) = profile.gpu {
        if gpu.vram_bytes >= 8 * 1024 * 1024 * 1024 {
            enabled_tiers.push(Tier::Standard);
        }
        if gpu.vram_bytes >= 24 * 1024 * 1024 * 1024 {
            enabled_tiers.push(Tier::Pro);
        }
        if gpu.vram_bytes >= 48 * 1024 * 1024 * 1024 {
            enabled_tiers.push(Tier::Max);
        }
    } else if profile.ram_bytes >= 32 * 1024 * 1024 * 1024 {
        enabled_tiers.push(Tier::Standard);
    }

    // Pro tier: GPU with 24GB VRAM or 64GB RAM
    if let Some(ref gpu) = profile.gpu {
        if gpu.vram_bytes >= 24 * 1024 * 1024 * 1024 {
            enabled_tiers.push(Tier::Pro);
        }
    } else if profile.ram_bytes >= 64 * 1024 * 1024 * 1024 {
        enabled_tiers.push(Tier::Pro);
    }

    // Max tier: GPU with 48GB VRAM or 128GB RAM
    if let Some(ref gpu) = profile.gpu {
        if gpu.vram_bytes >= 48 * 1024 * 1024 * 1024 {
            enabled_tiers.push(Tier::Max);
        }
    } else if profile.ram_bytes >= 128 * 1024 * 1024 * 1024 {
        enabled_tiers.push(Tier::Max);
    }

    let recommended_batch_size = match profile.gpu {
        Some(ref gpu) if gpu.vram_bytes >= 48 * 1024 * 1024 * 1024 => 64,
        Some(ref gpu) if gpu.vram_bytes >= 24 * 1024 * 1024 * 1024 => 32,
        Some(ref gpu) if gpu.vram_bytes >= 8 * 1024 * 1024 * 1024 => 16,
        _ => 4,
    };

    let max_concurrent_requests = (profile.cpu.cores_logical / 2).max(1);

    TierConfiguration {
        enabled_tiers,
        recommended_batch_size,
        max_concurrent_requests,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cpu() {
        let cpu = detect_cpu().unwrap();
        assert!(!cpu.vendor.is_empty());
        assert!(!cpu.brand.is_empty());
        assert!(cpu.cores_physical > 0);
        assert!(cpu.cores_logical > 0);
        assert!(cpu.frequency_mhz > 0);
        assert!(!cpu.features.is_empty());
    }

    #[test]
    fn test_detect_ram() {
        let ram = detect_ram().unwrap();
        assert!(ram > 0);
    }

    #[test]
    fn test_determine_tiers() {
        let profile = HardwareProfile {
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
        };

        let tiers = determine_tiers(&profile);
        assert!(tiers.enabled_tiers.contains(&Tier::Nano));
        assert!(tiers.enabled_tiers.contains(&Tier::Standard));
        assert!(!tiers.enabled_tiers.contains(&Tier::Pro));
        assert!(!tiers.enabled_tiers.contains(&Tier::Max));
        assert_eq!(tiers.recommended_batch_size, 16);
        assert_eq!(tiers.max_concurrent_requests, 8);
    }

    #[test]
    fn test_determine_tiers_no_gpu() {
        let profile = HardwareProfile {
            cpu: CpuProfile {
                vendor: "x86".to_string(),
                brand: "Test CPU".to_string(),
                cores_physical: 16,
                cores_logical: 32,
                frequency_mhz: 3000,
                features: vec!["sse4.2".to_string(), "avx2".to_string()],
            },
            gpu: None,
            ram_bytes: 64 * 1024 * 1024 * 1024,
            ram_bandwidth_gbps: 50.0,
            disk_bandwidth_mbps: 500.0,
            disk_total_bytes: 500 * 1024 * 1024 * 1024,
            network_latency_ms: 50.0,
        };

        let tiers = determine_tiers(&profile);
        assert!(tiers.enabled_tiers.contains(&Tier::Nano));
        assert!(tiers.enabled_tiers.contains(&Tier::Standard));
        assert!(tiers.enabled_tiers.contains(&Tier::Pro));
        assert!(!tiers.enabled_tiers.contains(&Tier::Max));
        assert_eq!(tiers.recommended_batch_size, 4);
        assert_eq!(tiers.max_concurrent_requests, 16);
    }
}
