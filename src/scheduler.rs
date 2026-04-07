use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRequest {
    pub request_id: String,
    pub expert_ids: Vec<String>,
    pub input: Tensor,
    pub model_state: ModelState,
    pub priority: u32,
    pub tier: String,
    pub submission_time: Instant,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerResponse {
    pub request_id: String,
    pub status: SchedulerStatus,
    pub estimated_wait_time_ms: u64,
    pub assigned_worker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerStatus {
    Pending,
    Scheduled,
    Executing,
    Completed,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub tier: String,
    pub capacity: u32,
    pub available_capacity: u32,
    pub device_type: String,
    pub device_count: u32,
    pub memory_mb: u32,
    pub last_heartbeat: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub request_id: String,
    pub worker_id: String,
    pub resources: Vec<Resource>,
    pub allocation_time: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Resource {
    Cpu { cores: u32, memory_mb: u32 },
    Gpu { device_id: u32, memory_mb: u32 },
    Memory { size_mb: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingMetrics {
    pub total_requests: u64,
    pub pending_requests: u64,
    pub scheduled_requests: u64,
    pub executing_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub gpu_utilization: f32,
    pub cpu_utilization: f32,
}

pub struct Scheduler {
    config: SchedulerConfig,
    request_queue: Arc<RwLock<BinaryHeap<SchedulerRequest>>>,
    worker_registry: Arc<RwLock<HashMap<String, WorkerInfo>>>,
    resource_pool: Arc<RwLock<HashMap<String, Vec<Resource>>>>,
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    metrics: Arc<RwLock<SchedulingMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_queue_size: usize,
    pub default_priority: u32,
    pub timeout_seconds: u64,
    pub batch_size: u32,
    pub resource_overcommit_factor: f32,
    pub worker_heartbeat_timeout_seconds: u64,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            request_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            worker_registry: Arc::new(RwLock::new(HashMap::new())),
            resource_pool: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::wLock<HashMap<String, ResourceAllocation>>>,
            metrics: Arc::new(RwLock::new(SchedulingMetrics::new())),
        }
    }

    pub fn submit_request(&self, request: SchedulerRequest) -> AuriaResult<SchedulerResponse> {
        let mut queue = self.request_queue.write().unwrap();

        // Check queue size limit
        if queue.len() >= self.config.max_queue_size {
            return Err(AuriaError::SchedulerError("Queue is full".to_string()));
        }

        // Add request to priority queue
        queue.push(request);

        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_requests += 1;
        metrics.pending_requests += 1;

        Ok(SchedulerResponse {
            request_id: request.request_id.clone(),
            status: SchedulerStatus::Pending,
            estimated_wait_time_ms: self.estimate_wait_time(&request)?,
            assigned_worker: None,
        })
    }

    pub fn schedule_requests(&self) -> AuriaResult<Vec<SchedulerResponse>> {
        let mut responses = Vec::new();
        let mut queue = self.request_queue.write().unwrap();
        let mut allocations = self.allocations.write().unwrap();

        // Get available workers
        let workers = self.get_available_workers()?;
        if workers.is_empty() {
            return Ok(responses); // No workers available
        }

        // Process requests from queue
        while let Some(request) = queue.pop() {
            // Find suitable worker
            if let Some((worker_id, resources)) = self.find_suitable_worker(&request, &workers) {
                // Allocate resources
                let allocation = ResourceAllocation {
                    request_id: request.request_id.clone(),
                    worker_id: worker_id.clone(),
                    resources,
                    allocation_time: Instant::now(),
                };

                allocations.insert(request.request_id.clone(), allocation);

                // Update worker capacity
                self.update_worker_capacity(&worker_id, &resources)?;

                // Create response
                let response = SchedulerResponse {
                    request_id: request.request_id.clone(),
                    status: SchedulerStatus::Scheduled,
                    estimated_wait_time_ms: 0,
                    assigned_worker: Some(worker_id),
                };

                responses.push(response);

                // Update metrics
                let mut metrics = self.metrics.write().unwrap();
                metrics.pending_requests -= 1;
                metrics.scheduled_requests += 1;
            } else {
                // No suitable worker found, put request back in queue
                queue.push(request);
                break;
            }
        }

        Ok(responses)
    }

    pub fn register_worker(&self, worker_info: WorkerInfo) -> AuriaResult<()> {
        let mut registry = self.worker_registry.write().unwrap();
        registry.insert(worker_info.worker_id.clone(), worker_info);
        Ok(())
    }

    pub fn update_worker_heartbeat(&self, worker_id: &str) -> AuriaResult<()> {
        let mut registry = self.worker_registry.write().unwrap();
        if let Some(worker) = registry.get_mut(worker_id) {
            worker.last_heartbeat = Instant::now();
            Ok(())
        } else {
            Err(AuriaError::SchedulerError(format!("Worker {} not found", worker_id)))
        }
    }

    pub fn get_worker_status(&self) -> Vec<WorkerInfo> {
        let registry = self.worker_registry.read().unwrap();
        let mut workers: Vec<_> = registry.values().cloned().collect();

        // Sort by last heartbeat
        workers.sort_by(|a, b| b.last_heartbeat.cmp(&a.last_heartbeat));
        workers
    }

    pub fn get_metrics(&self) -> SchedulingMetrics {
        self.metrics.read().unwrap().clone()
    }

    fn get_available_workers(&self) -> AuriaResult<Vec<WorkerInfo>> {
        let registry = self.worker_registry.read().unwrap();
        let mut available_workers = Vec::new();

        let now = Instant::now();
        let timeout = Duration::from_secs(self.config.worker_heartbeat_timeout_seconds as u64);

        for worker in registry.values() {
            // Check if worker is alive
            if now.duration_since(worker.last_heartbeat) < timeout {
                // Check if worker has available capacity
                if worker.available_capacity > 0 {
                    available_workers.push(worker.clone());
                }
            }
        }

        Ok(available_workers)
    }

    fn find_suitable_worker(&self, request: &SchedulerRequest, workers: &[WorkerInfo]) -> Option<(String, Vec<Resource>)> {
        for worker in workers {
            // Check if worker supports the request tier
            if worker.tier != request.tier {
                continue;
            }

            // Check resource requirements
            let required_resources = self.calculate_required_resources(request, worker)?;
            if self.has_sufficient_resources(worker, &required_resources) {
                return Some((worker.worker_id.clone(), required_resources));
            }
        }
        None
    }

    fn calculate_required_resources(&self, request: &SchedulerRequest, worker: &WorkerInfo) -> AuriaResult<Vec<Resource>> {
        let mut resources = Vec::new();

        // Calculate memory requirements
        let input_memory_mb = (request.input.data.len() as u32 / 1024 / 1024) as u32;
        let kv_cache_memory_mb = request.model_state.kv_cache.iter()
            .map(|t| (t.data.len() as u32 / 1024 / 1024) as u32)
            .sum();
        let total_memory_mb = input_memory_mb + kv_cache_memory_mb + 100; // Add buffer

        resources.push(Resource::Memory { size_mb: total_memory_mb });

        // Calculate CPU/GPU requirements based on tier
        match request.tier.as_str() {
            "nano" => {
                resources.push(Resource::Cpu { cores: 1, memory_mb: 512 });
            }
            "standard" => {
                if worker.device_type == "gpu" {
                    resources.push(Resource::Gpu { device_id: 0, memory_mb: 4096 });
                } else {
                    resources.push(Resource::Cpu { cores: 4, memory_mb: 2048 });
                }
            }
            "pro" => {
                if worker.device_type == "gpu" {
                    resources.push(Resource::Gpu { device_id: 0, memory_mb: 12288 });
                } else {
                    resources.push(Resource::Cpu { cores: 8, memory_mb: 4096 });
                }
            }
            "max" => {
                if worker.device_type == "gpu" {
                    resources.push(Resource::Gpu { device_id: 0, memory_mb: 24576 });
                } else {
                    resources.push(Resource::Cpu { cores: 16, memory_mb: 8192 });
                }
            }
            _ => {
                return Err(AuriaError::SchedulerError(format!("Unknown tier: {}", request.tier)));
            }
        }

        Ok(resources)
    }

    fn has_sufficient_resources(&self, worker: &WorkerInfo, required_resources: &[Resource]) -> bool {
        let mut total_cpu_cores = 0;
        let mut total_memory_mb = 0;
        let mut total_gpu_memory_mb = 0;

        for resource in required_resources {
            match resource {
                Resource::Cpu { cores, memory_mb } => {
                    total_cpu_cores += cores;
                    total_memory_mb += memory_mb;
                }
                Resource::Gpu { memory_mb, .. } => {
                    total_gpu_memory_mb += memory_mb;
                }
                Resource::Memory { size_mb } => {
                    total_memory_mb += size_mb;
                }
            }
        }

        // Check if worker has sufficient resources
        if total_cpu_cores > 0 && total_cpu_cores > worker.capacity {
            return false;
        }

        if total_memory_mb > worker.memory_mb {
            return false;
        }

        if total_gpu_memory_mb > 0 && worker.device_type != "gpu" {
            return false;
        }

        true
    }

    fn update_worker_capacity(&self, worker_id: &str, resources: &[Resource]) -> AuriaResult<()> {
        let mut registry = self.worker_registry.write().unwrap();
        if let Some(worker) = registry.get_mut(worker_id) {
            for resource in resources {
                match resource {
                    Resource::Cpu { cores, .. } => {
                        worker.available_capacity -= cores;
                    }
                    Resource::Gpu { .. } => {
                        // GPU capacity handling
                    }
                    Resource::Memory { size_mb } => {
                        worker.memory_mb -= size_mb;
                    }
                }
            }
            Ok(())
        } else {
            Err(AuriaError::SchedulerError(format!("Worker {} not found", worker_id)))
        }
    }

    fn estimate_wait_time(&self, request: &SchedulerRequest) -> AuriaResult<u64> {
        // Simple estimation based on queue length and average processing time
        let queue = self.request_queue.read().unwrap();
        let queue_length = queue.len() as u64;

        // Get average processing time from metrics
        let metrics = self.metrics.read().unwrap();
        let avg_processing_time_ms = if metrics.total_requests > 0 {
            metrics.average_execution_time_ms as u64
        } else {
            100 // Default 100ms
        };

        Ok(queue_length * avg_processing_time_ms)
    }

    pub fn cleanup_completed_requests(&self) {
        let mut allocations = self.allocations.write().unwrap();
        let mut completed_ids = Vec::new();

        for (request_id, allocation) in allocations.iter() {
            // Check if request has completed (in real implementation, this would be checked with worker)
            if Instant::now().duration_since(allocation.allocation_time) > Duration::from_secs(3600) {
                completed_ids.push(request_id.clone());
            }
        }

        for request_id in completed_ids {
            allocations.remove(&request_id);
        }
    }
}

impl SchedulerRequest {
    pub fn new(
        expert_ids: Vec<String>,
        input: Tensor,
        model_state: ModelState,
        priority: u32,
        tier: String,
    ) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            expert_ids,
            input,
            model_state,
            priority,
            tier,
            submission_time: Instant::now(),
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl SchedulerResponse {
    pub fn new(request_id: String, status: SchedulerStatus) -> Self {
        Self {
            request_id,
            status,
            estimated_wait_time_ms: 0,
            assigned_worker: None,
        }
    }
}

impl WorkerInfo {
    pub fn new(
        worker_id: String,
        tier: String,
        capacity: u32,
        device_type: String,
        device_count: u32,
        memory_mb: u32,
    ) -> Self {
        Self {
            worker_id,
            tier,
            capacity,
            available_capacity: capacity,
            device_type,
            device_count,
            memory_mb,
            last_heartbeat: Instant::now(),
        }
    }
}

impl ResourceAllocation {
    pub fn new(
        request_id: String,
        worker_id: String,
        resources: Vec<Resource>,
    ) -> Self {
        Self {
            request_id,
            worker_id,
            resources,
            allocation_time: Instant::now(),
        }
    }
}

impl SchedulingMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            pending_requests: 0,
            scheduled_requests: 0,
            executing_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            average_wait_time_ms: 0.0,
            average_execution_time_ms: 0.0,
            gpu_utilization: 0.0,
            cpu_utilization: 0.0,
        }
    }

    pub fn update_wait_time(&mut self, wait_time_ms: u64) {
        self.average_wait_time_ms = (self.average_wait_time_ms * (self.total_requests - 1) as f64 + wait_time_ms as f64) / self.total_requests as f64;
    }

    pub fn update_execution_time(&mut self, execution_time_ms: u64) {
        self.average_execution_time_ms = (self.average_execution_time_ms * (self.total_requests - 1) as f64 + execution_time_ms as f64) / self.total_requests as f64;
    }

    pub fn update_utilization(&mut self, gpu_utilization: f32, cpu_utilization: f32) {
        self.gpu_utilization = gpu_utilization;
        self.cpu_utilization = cpu_utilization;
    }
}

impl Ord for SchedulerRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
            .then_with(|| self.submission_time.cmp(&other.submission_time))
    }
}

impl PartialOrd for SchedulerRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SchedulerRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submission_time == other.submission_time
    }
}

impl Eq for SchedulerRequest {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_basic_operations() {
        let config = SchedulerConfig {
            max_queue_size: 100,
            default_priority: 5,
            timeout_seconds: 300,
            batch_size: 8,
            resource_overcommit_factor: 1.2,
            worker_heartbeat_timeout_seconds: 60,
        };

        let scheduler = Scheduler::new(config);

        // Create test request
        let request = SchedulerRequest::new(
            vec!["expert_1".to_string(), "expert_2".to_string()],
            Tensor {
                data: vec![1, 2, 3, 4, 5],
                shape: vec![5],
                dtype: TensorDType::Int8,
                device: ExecutionDevice::Cpu,
                requires_grad: false,
            },
            ModelState {
                position: 0,
                kv_cache: vec![],
                attention_mask: Tensor {
                    data: vec![1, 1, 1, 1, 1],
                    shape: vec![5],
                    dtype: TensorDType::Int8,
                    device: ExecutionDevice::Cpu,
                    requires_grad: false,
                },
                past_key_values: None,
                sequence_length: 5,
            },
            10, // High priority
            "standard".to_string(),
        );

        // Submit request
        let response = scheduler.submit_request(request.clone()).unwrap();
        assert_eq!(response.request_id, request.request_id);
        assert_eq!(response.status, SchedulerStatus::Pending);
        assert!(response.estimated_wait_time_ms > 0);

        // Schedule requests
        let responses = scheduler.schedule_requests().unwrap();
        assert!(!responses.is_empty());

        // Register worker
        let worker = WorkerInfo::new(
            "worker_1".to_string(),
            "standard".to_string(),
            4,
            "gpu".to_string(),
            1,
            8192,
        );
        scheduler.register_worker(worker).unwrap();

        // Update worker heartbeat
        scheduler.update_worker_heartbeat("worker_1").unwrap();

        // Get worker status
        let workers = scheduler.get_worker_status();
        assert!(!workers.is_empty());

        // Get metrics
        let metrics = scheduler.get_metrics();
        assert!(metrics.total_requests > 0);
    }

    #[test]
    fn test_resource_allocation() {
        let config = SchedulerConfig {
            max_queue_size: 100,
            default_priority: 5,
            timeout_seconds: 300,
            batch_size: 8,
            resource_overcommit_factor: 1.2,
            worker_heartbeat_timeout_seconds: 60,
        };

        let scheduler = Scheduler::new(config);

        // Create test request
        let request = SchedulerRequest::new(
            vec!["expert_1".to_string()],
            Tensor {
                data: vec![1; 1024 * 1024], // 1MB
                shape: vec![1024],
                dtype: TensorDType::Int8,
                device: ExecutionDevice::Cpu,
                requires_grad: false,
            },
            ModelState {
                position: 0,
                kv_cache: vec![],
                attention_mask: Tensor {
                    data: vec![1; 1024],
                    shape: vec![1024],
                    dtype: TensorDType::Int8,
                    device: ExecutionDevice::Cpu,
                    requires_grad: false,
                },
                past_key_values: None,
                sequence_length: 1024,
            },
            5,
            "nano".to_string(),
        );

        // Register worker
        let worker = WorkerInfo::new(
            "worker_nano".to_string(),
            "nano".to_string(),
            1,
            "cpu".to_string(),
            1,
            2048,
        );
        scheduler.register_worker(worker).unwrap();

        // Schedule request
        let responses = scheduler.schedule_requests().unwrap();
        assert!(!responses.is_empty());

        // Check resource allocation
        let response = &responses[0];
        assert!(response.assigned_worker.is_some());
        assert_eq!(response.status, SchedulerStatus::Scheduled);
    }

    #[test]
    fn test_priority_queue() {
        let config = SchedulerConfig {
            max_queue_size: 100,
            default_priority: 5,
            timeout_seconds: 300,
            batch_size: 8,
            resource_overcommit_factor: 1.2,
            worker_heartbeat_timeout_seconds: 60,
        };

        let scheduler = Scheduler::new(config);

        // Create requests with different priorities
        let high_priority_request = SchedulerRequest::new(
            vec!["expert_high".to_string()],
            Tensor { data: vec![1; 1024], shape: vec![1024], dtype: TensorDType::Int8, device: ExecutionDevice::Cpu, requires_grad: false },
            ModelState { position: 0, kv_cache: vec![], attention_mask: Tensor { data: vec![1; 1024], shape: vec![1024], dtype: TensorDType::Int8, device: ExecutionDevice::Cpu, requires_grad: false }, past_key_values: None, sequence_length: 1024 },
            10, // High priority
            "nano".to_string(),
        );

        let low_priority_request = SchedulerRequest::new(
            vec!["expert_low".to_string()],
            Tensor { data: vec![2; 1024], shape: vec![1024], dtype: TensorDType::Int8, device: ExecutionDevice::Cpu, requires_grad: false },
            ModelState { position: 0, kv_cache: vec![], attention_mask: Tensor { data: vec![2; 1024], shape: vec![1024], dtype: TensorDType::Int8, device: ExecutionDevice::Cpu, requires_grad: false }, past_key_values: None, sequence_length: 1024 },
            1, // Low priority
            "nano".to_string(),
        );

        // Submit requests
        scheduler.submit_request(high_priority_request.clone()).unwrap();
        scheduler.submit_request(low_priority_request.clone()).unwrap();

        // Schedule requests - high priority should come first
        let responses = scheduler.schedule_requests().unwrap();
        assert!(!responses.is_empty());
        assert_eq!(responses[0].request_id, high_priority_request.request_id);
    }
}

pub use self::Scheduler;
pub use self::SchedulerRequest;
pub use self::SchedulerResponse;
pub use self::SchedulerStatus;
pub use self::WorkerInfo;
pub use self::ResourceAllocation;
pub use self::Resource;
pub use self::SchedulingMetrics;
pub use self::SchedulerConfig;