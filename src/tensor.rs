use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    pub dtype: TensorDType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDType {
    FP16,
    FP8,
    INT8,
    INT4,
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDType::FP16 => write!(f, "FP16"),
            TensorDType::FP8 => write!(f, "FP8"),
            TensorDType::INT8 => write!(f, "INT8"),
            TensorDType::INT4 => write!(f, "INT4"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLayout {
    pub offset: u64,
    pub stride: u32,
    pub shape: Vec<u32>,
}

impl TensorLayout {
    pub fn new(shape: Vec<u32>) -> Self {
        Self {
            offset: 0,
            stride: 1,
            shape,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOperation {
    pub operation: String,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
}

pub type TensorResult<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorError {
    InvalidShape,
    InvalidDtype,
    MemoryError,
    ComputationError,
    InvalidOperation,
}
