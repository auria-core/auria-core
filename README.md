# auria-core

The foundational crate for AURIA Runtime Core. Contains all core types, data structures, and error types used throughout the runtime.

## Types

- `Tensor` - Multi-dimensional array with dtype support (FP16, FP8, INT8, INT4)
- `Shard` - Licensed computational unit containing tensor data
- `Expert` - Assembled expert composed of multiple shards
- `HardwareProfile` - Node hardware capabilities
- `Tier` - Execution tier (Nano, Standard, Pro, Max)
- `License` - Shard access authorization
- `UsageReceipt` - Settlement proof

## Error Types

All runtime errors are unified under `AuriaError`:

```rust
pub enum AuriaError {
    ShardNotFound(ShardId),
    ExpertNotFound(ExpertId),
    LicenseInvalid(ShardId),
    InsufficientHardware(Tier),
    StorageError(String),
    ExecutionError(String),
    NetworkError(String),
    // ...
}
```

## Usage

```rust
use auria_core::{Tensor, TensorDType, Tier};

let tensor = Tensor {
    data: vec![1, 2, 3, 4],
    shape: vec![4],
    dtype: TensorDType::FP16,
};
```

## Dependencies

None - this is the foundational crate that all other crates depend on.
