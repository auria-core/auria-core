use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RuntimeVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl RuntimeVersion {
    pub fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    pub fn is_compatible(&self, required: &RuntimeVersion) -> bool {
        self.major == required.major && self.minor >= required.minor
    }

    pub fn from_str(version: &str) -> Option<Self> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        let major = parts[0].parse::<u16>().ok()?;
        let minor = parts[1].parse::<u16>().ok()?;
        let patch = parts[2].parse::<u16>().ok()?;

        Some(Self { major, minor, patch })
    }
}

impl fmt::Display for RuntimeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub runtime_version: RuntimeVersion,
    pub api_version: RuntimeVersion,
    pub git_commit: Option<String>,
    pub build_date: String,
    pub rust_version: String,
    pub features: Vec<String>,
}

impl VersionInfo {
    pub fn new() -> Self {
        Self {
            runtime_version: RuntimeVersion::current(),
            api_version: RuntimeVersion::current(),
            git_commit: option_env!("GIT_COMMIT").map(|s| s.to_string()),
            build_date: chrono::Utc::now().to_rfc3339(),
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            features: vec![
                #[cfg(feature = "gpu")]
                "gpu".to_string(),
                #[cfg(not(feature = "gpu"))]
                "cpu-only".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ProtocolVersion {
    pub fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    pub fn is_compatible(&self, required: &ProtocolVersion) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    pub runtime_version: RuntimeVersion,
    pub protocol_version: ProtocolVersion,
    pub supported_features: Vec<String>,
    pub minimum_requirements: HardwareRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub min_ram_gb: u32,
    pub min_storage_gb: u32,
    pub min_cpu_cores: u32,
    pub gpu_required: bool,
    pub min_gpu_vram_gb: Option<u32>,
}

impl HardwareRequirements {
    pub fn default() -> Self {
        Self {
            min_ram_gb: 4,
            min_storage_gb: 10,
            min_cpu_cores: 2,
            gpu_required: false,
            min_gpu_vram_gb: None,
        }
    }

    pub fn for_tier(tier: &str) -> Self {
        match tier {
            "nano" => Self {
                min_ram_gb: 4,
                min_storage_gb: 5,
                min_cpu_cores: 2,
                gpu_required: false,
                min_gpu_vram_gb: None,
            },
            "standard" => Self {
                min_ram_gb: 8,
                min_storage_gb: 20,
                min_cpu_cores: 4,
                gpu_required: true,
                min_gpu_vram_gb: Some(4),
            },
            "pro" => Self {
                min_ram_gb: 16,
                min_storage_gb: 50,
                min_cpu_cores: 8,
                gpu_required: true,
                min_gpu_vram_gb: Some(12),
            },
            "max" => Self {
                min_ram_gb: 64,
                min_storage_gb: 200,
                min_cpu_cores: 16,
                gpu_required: true,
                min_gpu_vram_gb: Some(24),
            },
            _ => Self::default(),
        }
    }
}