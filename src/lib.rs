//! AURIA Runtime Core - Foundational types and data structures
//!
//! This crate defines the core types, traits, and error handling that all other AURIA modules depend on.
//!
//! # Core Concepts
//!
//! - **Tensor**: The fundamental data structure for neural network computations
//! - **Shard**: The smallest unit of intelligence storage and ownership
//! - **Expert**: Deterministic assembly of shards into executable intelligence
//! - **RuntimeVersion**: Version tracking for the AURIA Runtime Core
//!
//! # Error Handling
//!
//! All AURIA modules use the unified `AuriaError` type defined in this crate.
//!
//! # Feature Flags
//!
//! - `gpu`: Enables GPU-related types and functionality (currently unused in core)

pub mod error;
pub mod expert;
pub mod license;
pub mod router;
pub mod settlement;
pub mod shard;
pub mod tensor;
pub mod version;

pub use error::{AuriaError, AuriaResult, Tier};
pub use expert::Expert;
pub use license::License;
pub use router::{Router as RoutingRouter, RoutingDecision};
pub use settlement::{RequestId, UsageReceipt, UsageStats};
pub use shard::{ExpertId, Hash, LicenseHash, PublicKey, Shard, ShardId, Signature};
pub use tensor::{Tensor, TensorDType, TensorLayout};
pub use version::RuntimeVersion;
