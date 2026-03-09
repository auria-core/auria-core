// File: hardware_benchmark.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Hardware benchmarking for AURIA Runtime Core.
//     Provides performance benchmarking for accurate tier assignment and
//     performance monitoring of hardware capabilities.
//
use auria_core::{AuriaError, AuriaResult, Tier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub score: f64,
    pub units: String,
    pub duration: Duration,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub version: String,
    pub results: Vec<BenchmarkResult>,
    pub total_score: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub cpu_benchmarks: Vec<BenchmarkResult>,
    pub gpu_benchmarks: Vec<BenchmarkResult>,
    pub memory_benchmarks: Vec<BenchmarkResult>,
    pub storage_benchmarks: Vec<BenchmarkResult>,
    pub network_benchmarks: Vec<BenchmarkResult>,
    pub overall_score: f64,
    pub tier_recommendation: Tier,
}

pub struct HardwareBenchmarker {
    cpu_cores: u32,
    cpu_frequency: u64,
    has_gpu: bool,
}

impl HardwareBenchmarker {
    pub fn new(cpu_cores: u32, cpu_frequency: u64, has_gpu: bool) -> Self {
        Self {
            cpu_cores,
            cpu_frequency,
            has_gpu,
        }
    }

    pub fn run_benchmark_suite(&self) -> AuriaResult<BenchmarkSuite> {
        let start_time = Instant::now();
        let mut results = Vec::new();

        // CPU Benchmarks
        if self.cpu_cores > 0 {
            results.push(self.benchmark_cpu_int_float()?);
            results.push(self.benchmark_cpu_prime()?);
            results.push(self.benchmark_cpu_sort()?);
        }

        // GPU Benchmarks (if available)
        if self.has_gpu {
            results.push(self.benchmark_gpu_matrix_multiplication()?);
            results.push(self.benchmark_gpu_convolution()?);
        }

        // Memory Benchmarks
        results.push(self.benchmark_memory_bandwidth()?);
        results.push(self.benchmark_memory_latency()?);

        // Storage Benchmarks
        results.push(self.benchmark_storage_read()?);
        results.push(self.benchmark_storage_write()?);

        // Network Benchmarks
        results.push(self.benchmark_network_latency()?);
        results.push(self.benchmark_network_bandwidth()?);

        let total_score = results.iter().map(|r| r.score).sum::<f64>();
        let duration = start_time.elapsed();

        let suite = BenchmarkSuite {
            name: "AURIA Hardware Benchmark Suite".to_string(),
            version: "1.0.0".to_string(),
            results,
            total_score,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        log::info!("Benchmark suite completed in {:?}. Total score: {:.2}", duration, total_score);

        Ok(suite)
    }

    pub fn analyze_performance(&self, suite: &BenchmarkSuite) -> PerformanceProfile {
        let mut profile = PerformanceProfile {
            cpu_benchmarks: Vec::new(),
            gpu_benchmarks: Vec::new(),
            memory_benchmarks: Vec::new(),
            storage_benchmarks: Vec::new(),
            network_benchmarks: Vec::new(),
            overall_score: suite.total_score,
            tier_recommendation: Tier::Nano, // Default
        };

        // Categorize results
        for result in &suite.results {
            match result.benchmark_name.as_str() {
                "cpu_int_float" | "cpu_prime" | "cpu_sort" => {
                    profile.cpu_benchmarks.push(result.clone());
                }
                "gpu_matrix_mult" | "gpu_convolution" => {
                    profile.gpu_benchmarks.push(result.clone());
                }
                "memory_bandwidth" | "memory_latency" => {
                    profile.memory_benchmarks.push(result.clone());
                }
                "storage_read" | "storage_write" => {
                    profile.storage_benchmarks.push(result.clone());
                }
                "network_latency" | "network_bandwidth" => {
                    profile.network_benchmarks.push(result.clone());
                }
                _ => {}
            }
        }

        // Determine tier recommendation based on scores
        profile.tier_recommendation = self.determine_tier_recommendation(&profile);

        profile
    }

    fn benchmark_cpu_int_float(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let iterations = 10_000_000;
        let mut result = 0.0;

        for i in 0..iterations {
            let a = (i as f64).sin();
            let b = (i as f64).cos();
            result += a * b;
        }

        let duration = start_time.elapsed();
        let score = (iterations as f64) / duration.as_secs_f64();

        Ok(BenchmarkResult {
            benchmark_name: "cpu_int_float".to_string(),
            score,
            units: "ops/sec".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_cpu_prime(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let limit = 100_000;
        let mut primes = Vec::new();

        for num in 2..=limit {
            let mut is_prime = true;
            for i in 2..=((num as f64).sqrt() as u64) {
                if num % i == 0 {
                    is_prime = false;
                    break;
                }
            }
            if is_prime {
                primes.push(num);
            }
        }

        let duration = start_time.elapsed();
        let score = (limit as f64) / duration.as_secs_f64();

        Ok(BenchmarkResult {
            benchmark_name: "cpu_prime".to_string(),
            score,
            units: "nums/sec".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_cpu_sort(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let size = 100_000;
        let mut data: Vec<u32> = (0..size).map(|x| rand::random::<u32>() % size).collect();

        data.sort();

        let duration = start_time.elapsed();
        let score = (size as f64) / duration.as_secs_f64();

        Ok(BenchmarkResult {
            benchmark_name: "cpu_sort".to_string(),
            score,
            units: "elems/sec".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_gpu_matrix_multiplication(&self) -> AuriaResult<BenchmarkResult> {
        // This is a placeholder - actual GPU benchmarking would require CUDA/Metal/ROCm
        let start_time = Instant::now();
        let size = 1024;
        let mut a = vec![0.0; size * size];
        let mut b = vec![0.0; size * size];
        let mut c = vec![0.0; size * size];

        // Initialize matrices
        for i in 0..(size * size) {
            a[i] = rand::random::<f64>();
            b[i] = rand::random::<f64>();
        }

        // Matrix multiplication (naive implementation)
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    sum += a[i * size + k] * b[k * size + j];
                }
                c[i * size + j] = sum;
            }
        }

        let duration = start_time.elapsed();
        let score = (size * size * size) as f64 / duration.as_secs_f64();

        Ok(BenchmarkResult {
            benchmark_name: "gpu_matrix_mult".to_string(),
            score,
            units: "flops".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_gpu_convolution(&self) -> AuriaResult<BenchmarkResult> {
        // Placeholder for GPU convolution benchmark
        let start_time = Instant::now();
        let size = 512;
        let kernel_size = 3;
        let mut image = vec![0.0; size * size];
        let mut output = vec![0.0; size * size];

        // Initialize image
        for i in 0..(size * size) {
            image[i] = rand::random::<f64>();
        }

        // Convolution (naive implementation)
        for y in 1..(size - 1) {
            for x in 1..(size - 1) {
                let mut sum = 0.0;
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let image_x = x + kx - kernel_size / 2;
                        let image_y = y + ky - kernel_size / 2;
                        sum += image[image_y * size + image_x];
                    }
                }
                output[y * size + x] = sum / (kernel_size * kernel_size) as f64;
            }
        }

        let duration = start_time.elapsed();
        let score = (size * size) as f64 / duration.as_secs_f64();

        Ok(BenchmarkResult {
            benchmark_name: "gpu_convolution".to_string(),
            score,
            units: "pixels/sec".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_memory_bandwidth(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let size = 100_000_000; // 100MB
        let mut buffer = vec![0u8; size];

        // Write test
        for i in 0..size {
            buffer[i] = (i % 256) as u8;
        }

        let write_duration = start_time.elapsed();

        // Read test
        let mut sum = 0u64;
        for i in 0..size {
            sum += buffer[i] as u64;
        }

        let read_duration = start_time.elapsed();
        let total_duration = start_time.elapsed();

        let bandwidth = (size as f64 * 2.0) / (total_duration.as_secs_f64() * 1024.0 * 1024.0); // MB/s

        Ok(BenchmarkResult {
            benchmark_name: "memory_bandwidth".to_string(),
            score: bandwidth,
            units: "MB/s".to_string(),
            duration: total_duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_memory_latency(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let iterations = 1_000_000;
        let mut buffer = vec![0u8; 1024 * 1024]; // 1MB

        let mut sum = 0u64;
        for _ in 0..iterations {
            for i in 0..buffer.len() {
                sum += buffer[i] as u64;
            }
        }

        let duration = start_time.elapsed();
        let latency = (duration.as_secs_f64() * 1_000_000_000.0) / (iterations as f64 * buffer.len() as f64); // ns per access

        Ok(BenchmarkResult {
            benchmark_name: "memory_latency".to_string(),
            score: latency,
            units: "ns/access".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_storage_read(&self) -> AuriaResult<BenchmarkResult> {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let file_path = temp_file.path();

        // Create a test file
        let file_size = 100 * 1024 * 1024; // 100MB
        let mut file = std::fs::File::create(file_path).unwrap();
        let data: Vec<u8> = (0..file_size).map(|x| (x % 256) as u8).collect();
        file.write_all(&data).unwrap();

        let start_time = Instant::now();
        let mut read_data = vec![0u8; file_size];
        let mut file = std::fs::File::open(file_path).unwrap();
        file.read_exact(&mut read_data).unwrap();
        let duration = start_time.elapsed();

        let bandwidth = (file_size as f64) / (duration.as_secs_f64() * 1024.0 * 1024.0); // MB/s

        Ok(BenchmarkResult {
            benchmark_name: "storage_read".to_string(),
            score: bandwidth,
            units: "MB/s".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_storage_write(&self) -> AuriaResult<BenchmarkResult> {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let file_path = temp_file.path();
        let file_size = 100 * 1024 * 1024; // 100MB
        let data: Vec<u8> = (0..file_size).map(|x| (x % 256) as u8).collect();

        let start_time = Instant::now();
        let mut file = std::fs::File::create(file_path).unwrap();
        file.write_all(&data).unwrap();
        let duration = start_time.elapsed();

        let bandwidth = (file_size as f64) / (duration.as_secs_f64() * 1024.0 * 1024.0); // MB/s

        Ok(BenchmarkResult {
            benchmark_name: "storage_write".to_string(),
            score: bandwidth,
            units: "MB/s".to_string(),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn benchmark_network_latency(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let endpoint = "https://www.google.com";

        let response = reqwest::blocking::get(endpoint);
        let duration = start_time.elapsed();

        if response.is_ok() {
            let latency_ms = duration.as_secs_f64() * 1000.0;
            Ok(BenchmarkResult {
                benchmark_name: "network_latency".to_string(),
                score: latency_ms,
                units: "ms".to_string(),
                duration,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            })
        } else {
            Err(AuriaError::NetworkError(format!("Failed to measure network latency: {}", response.err().unwrap_or("Unknown error"))))
        }
    }

    fn benchmark_network_bandwidth(&self) -> AuriaResult<BenchmarkResult> {
        let start_time = Instant::now();
        let endpoint = "https://speed.hetzner.de/100MB.bin";

        let response = reqwest::blocking::get(endpoint);
        let duration = start_time.elapsed();

        if let Ok(response) = response {
            let file_size = response.content_length().unwrap_or(100 * 1024 * 1024); // Default 100MB
            let bandwidth = (file_size as f64) / (duration.as_secs_f64() * 1024.0 * 1024.0); // MB/s

            Ok(BenchmarkResult {
                benchmark_name: "network_bandwidth".to_string(),
                score: bandwidth,
                units: "MB/s".to_string(),
                duration,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            })
        } else {
            Err(AuriaError::NetworkError(format!("Failed to measure network bandwidth: {}", response.err().unwrap_or("Unknown error"))))
        }
    }

    fn determine_tier_recommendation(&self, profile: &PerformanceProfile) -> Tier {
        // Calculate weighted scores for each tier
        let mut tier_scores = HashMap::new();

        // Nano tier requirements
        let nano_score = self.calculate_tier_score(profile, Tier::Nano);
        tier_scores.insert(Tier::Nano, nano_score);

        // Standard tier requirements
        let standard_score = self.calculate_tier_score(profile, Tier::Standard);
        tier_scores.insert(Tier::Standard, standard_score);

        // Pro tier requirements
        let pro_score = self.calculate_tier_score(profile, Tier::Pro);
        tier_scores.insert(Tier::Pro, pro_score);

        // Max tier requirements
        let max_score = self.calculate_tier_score(profile, Tier::Max);
        tier_scores.insert(Tier::Max, max_score);

        // Find the highest scoring tier that meets minimum requirements
        let mut recommended_tier = Tier::Nano;
        let mut highest_score = 0.0;

        for (tier, score) in tier_scores {
            if score > highest_score && score >= 0.5 { // Minimum 50% of requirements
                recommended_tier = tier;
                highest_score = score;
            }
        }

        recommended_tier
    }

    fn calculate_tier_score(&self, profile: &PerformanceProfile, tier: Tier) -> f64 {
        let mut score = 0.0;
        let mut requirements_met = 0;
        let mut total_requirements = 0;

        match tier {
            Tier::Nano => {
                // CPU requirements
                total_requirements += 1;
                if profile.cpu_benchmarks.len() > 0 {
                    let avg_cpu_score: f64 = profile.cpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.cpu_benchmarks.len() as f64;
                    if avg_cpu_score >= 1_000_000.0 { // 1 million ops/sec
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // Memory requirements
                total_requirements += 1;
                if profile.memory_benchmarks.len() > 0 {
                    let avg_memory_score: f64 = profile.memory_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.memory_benchmarks.len() as f64;
                    if avg_memory_score >= 5_000.0 { // 5 GB/s
                        requirements_met += 1;
                        score += 1.0;
                    }
                }
            }
            Tier::Standard => {
                // CPU requirements
                total_requirements += 1;
                if profile.cpu_benchmarks.len() > 0 {
                    let avg_cpu_score: f64 = profile.cpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.cpu_benchmarks.len() as f64;
                    if avg_cpu_score >= 5_000_000.0 { // 5 million ops/sec
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // GPU requirements (if available)
                total_requirements += 1;
                if profile.gpu_benchmarks.len() > 0 {
                    let avg_gpu_score: f64 = profile.gpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.gpu_benchmarks.len() as f64;
                    if avg_gpu_score >= 1_000_000_000.0 { // 1 billion flops
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // Memory requirements
                total_requirements += 1;
                if profile.memory_benchmarks.len() > 0 {
                    let avg_memory_score: f64 = profile.memory_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.memory_benchmarks.len() as f64;
                    if avg_memory_score >= 20_000.0 { // 20 GB/s
                        requirements_met += 1;
                        score += 1.0;
                    }
                }
            }
            Tier::Pro => {
                // CPU requirements
                total_requirements += 1;
                if profile.cpu_benchmarks.len() > 0 {
                    let avg_cpu_score: f64 = profile.cpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.cpu_benchmarks.len() as f64;
                    if avg_cpu_score >= 20_000_000.0 { // 20 million ops/sec
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // GPU requirements (if available)
                total_requirements += 1;
                if profile.gpu_benchmarks.len() > 0 {
                    let avg_gpu_score: f64 = profile.gpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.gpu_benchmarks.len() as f64;
                    if avg_gpu_score >= 5_000_000_000.0 { // 5 billion flops
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // Memory requirements
                total_requirements += 1;
                if profile.memory_benchmarks.len() > 0 {
                    let avg_memory_score: f64 = profile.memory_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.memory_benchmarks.len() as f64;
                    if avg_memory_score >= 50_000.0 { // 50 GB/s
                        requirements_met += 1;
                        score += 1.0;
                    }
                }
            }
            Tier::Max => {
                // CPU requirements
                total_requirements += 1;
                if profile.cpu_benchmarks.len() > 0 {
                    let avg_cpu_score: f64 = profile.cpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.cpu_benchmarks.len() as f64;
                    if avg_cpu_score >= 50_000_000.0 { // 50 million ops/sec
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // GPU requirements (if available)
                total_requirements += 1;
                if profile.gpu_benchmarks.len() > 0 {
                    let avg_gpu_score: f64 = profile.gpu_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.gpu_benchmarks.len() as f64;
                    if avg_gpu_score >= 20_000_000_000.0 { // 20 billion flops
                        requirements_met += 1;
                        score += 1.0;
                    }
                }

                // Memory requirements
                total_requirements += 1;
                if profile.memory_benchmarks.len() > 0 {
                    let avg_memory_score: f64 = profile.memory_benchmarks.iter().map(|r| r.score).sum::<f64>() / profile.memory_benchmarks.len() as f64;
                    if avg_memory_score >= 100_000.0 { // 100 GB/s
                        requirements_met += 1;
                        score += 1.0;
                    }
                }
            }
        }

        if total_requirements > 0 {
            score = (score / total_requirements as f64) * (requirements_met as f64 / total_requirements as f64);
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite() {
        let benchmarker = HardwareBenchmarker::new(8, 3000, true);
        let suite = benchmarker.run_benchmark_suite().unwrap();

        assert!(!suite.results.is_empty());
        assert!(suite.total_score > 0.0);
        assert!(suite.duration.as_secs() < 300); // Should complete in under 5 minutes
    }

    #[test]
    fn test_performance_analysis() {
        let benchmarker = HardwareBenchmarker::new(8, 3000, true);
        let suite = benchmarker.run_benchmark_suite().unwrap();
        let profile = benchmarker.analyze_performance(&suite);

        assert!(profile.overall_score > 0.0);
        assert!(matches!(profile.tier_recommendation, Tier::Nano | Tier::Standard | Tier::Pro | Tier::Max));
        assert!(!profile.cpu_benchmarks.is_empty());
        assert!(!profile.memory_benchmarks.is_empty());
        assert!(!profile.storage_benchmarks.is_empty());
        assert!(!profile.network_benchmarks.is_empty());
    }

    #[test]
    fn test_tier_recommendation() {
        let benchmarker = HardwareBenchmarker::new(8, 3000, true);

        // Create mock performance profile for each tier
        let profiles = [
            (Tier::Nano, 1_000_000.0, 5_000.0, 500.0, 50.0), // CPU, Memory, Storage, Network scores
            (Tier::Standard, 5_000_000.0, 20_000.0, 1000.0, 30.0),
            (Tier::Pro, 20_000_000.0, 50_000.0, 2000.0, 20.0),
            (Tier::Max, 50_000_000.0, 100_000.0, 5000.0, 10.0),
        ];

        for (tier, cpu_score, memory_score, storage_score, network_score) in profiles {
            let profile = PerformanceProfile {
                cpu_benchmarks: vec![BenchmarkResult {
                    benchmark_name: "cpu_test".to_string(),
                    score: cpu_score,
                    units: "ops/sec".to_string(),
                    duration: Duration::from_secs(1),
                    timestamp: 0,
                }],
                gpu_benchmarks: Vec::new(),
                memory_benchmarks: vec![BenchmarkResult {
                    benchmark_name: "memory_test".to_string(),
                    score: memory_score,
                    units: "MB/s".to_string(),
                    duration: Duration::from_secs(1),
                    timestamp: 0,
                }],
                storage_benchmarks: vec![BenchmarkResult {
                    benchmark_name: "storage_test".to_string(),
                    score: storage_score,
                    units: "MB/s".to_string(),
                    duration: Duration::from_secs(1),
                    timestamp: 0,
                }],
                network_benchmarks: vec![BenchmarkResult {
                    benchmark_name: "network_test".to_string(),
                    score: network_score,
                    units: "ms".to_string(),
                    duration: Duration::from_secs(1),
                    timestamp: 0,
                }],
                overall_score: cpu_score + memory_score + storage_score + network_score,
                tier_recommendation: Tier::Nano, // Will be overwritten
            };

            let recommended_tier = benchmarker.determine_tier_recommendation(&profile);
            assert_eq!(recommended_tier, tier, "Expected tier {:?} for scores: CPU={}, Memory={}, Storage={}, Network={}", tier, cpu_score, memory_score, storage_score, network_score);
        }
    }
}
