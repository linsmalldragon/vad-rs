use std::path::Path;

use eyre::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};

#[cfg(feature = "coreml")]
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(feature = "openvino")]
use ort::execution_providers::OpenVINOExecutionProvider;

/// Creates an ONNX Runtime session with automatic GPU detection and CPU fallback.
///
/// The function tries GPU execution providers in priority order and automatically
/// falls back to CPU if GPU initialization fails. This is a graceful fallback
/// that handles cases where the model is not fully compatible with GPU providers.
///
/// Priority order:
/// 1. TensorRT (NVIDIA GPU - most optimized)
/// 2. CUDA (NVIDIA GPU)
/// 3. CoreML (Apple GPU/Neural Engine)
/// 4. DirectML (Windows AMD/Intel/NVIDIA)
/// 5. OpenVINO (Intel CPU/GPU/NPU)
/// 6. CPU (always available)
pub fn create_session<P: AsRef<Path>>(path: P) -> Result<Session> {
    // Try GPU first, fall back to CPU if it fails
    match try_create_session_with_gpu(path.as_ref()) {
        Ok(session) => Ok(session),
        Err(gpu_err) => {
            eprintln!(
                "GPU execution provider failed, falling back to CPU: {}",
                gpu_err
            );
            create_cpu_session(path.as_ref())
        }
    }
}

/// Try to create a session with GPU acceleration
fn try_create_session_with_gpu(path: &Path) -> Result<Session> {
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

    // Collect all available execution providers
    #[allow(unused_mut)]
    let mut providers: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();

    // NVIDIA TensorRT (highest priority for NVIDIA GPUs)
    #[cfg(feature = "tensorrt")]
    providers.push(TensorRTExecutionProvider::default().build());

    // NVIDIA CUDA
    #[cfg(feature = "cuda")]
    providers.push(CUDAExecutionProvider::default().build());

    // Apple CoreML (macOS/iOS) - Note: Some ONNX ops may not be supported
    #[cfg(feature = "coreml")]
    providers.push(
        CoreMLExecutionProvider::default()
            .with_compute_units(CoreMLComputeUnits::All)
            .with_model_format(CoreMLModelFormat::NeuralNetwork)
            .build(),
    );

    // Windows DirectML (AMD/Intel/NVIDIA on Windows)
    #[cfg(feature = "directml")]
    providers.push(DirectMLExecutionProvider::default().build());

    // Intel OpenVINO
    #[cfg(feature = "openvino")]
    providers.push(OpenVINOExecutionProvider::default().build());

    if providers.is_empty() {
        // No GPU providers available, use CPU
        return create_cpu_session(path);
    }

    // Register all providers - ONNX Runtime will use the first available
    let session = builder
        .with_execution_providers(providers)?
        .with_intra_threads(4)?
        .commit_from_file(path)?;

    Ok(session)
}

/// Create a CPU-only session (fallback)
fn create_cpu_session(path: &Path) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(path)?;
    Ok(session)
}

/// Creates a session with a specific execution provider preference.
///
/// Useful when you want to force a specific backend or test different providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// Automatic detection (try all available)
    Auto,
    /// Force CPU only
    Cpu,
    /// Apple CoreML (macOS/iOS)
    #[cfg(feature = "coreml")]
    CoreML,
    /// NVIDIA CUDA
    #[cfg(feature = "cuda")]
    Cuda,
    /// NVIDIA TensorRT
    #[cfg(feature = "tensorrt")]
    TensorRT,
    /// Windows DirectML
    #[cfg(feature = "directml")]
    DirectML,
    /// Intel OpenVINO
    #[cfg(feature = "openvino")]
    OpenVINO,
}

pub fn create_session_with_provider<P: AsRef<Path>>(
    path: P,
    provider: ExecutionProvider,
) -> Result<Session> {
    match provider {
        ExecutionProvider::Auto => create_session(path),
        ExecutionProvider::Cpu => create_cpu_session(path.as_ref()),
        #[cfg(feature = "coreml")]
        ExecutionProvider::CoreML => {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CoreMLExecutionProvider::default()
                    .with_compute_units(CoreMLComputeUnits::All)
                    .with_model_format(CoreMLModelFormat::NeuralNetwork)
                    .build()])?
                .with_intra_threads(4)?
                .commit_from_file(path.as_ref())?;
            Ok(session)
        }
        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda => {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .with_intra_threads(4)?
                .commit_from_file(path.as_ref())?;
            Ok(session)
        }
        #[cfg(feature = "tensorrt")]
        ExecutionProvider::TensorRT => {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([TensorRTExecutionProvider::default().build()])?
                .with_intra_threads(4)?
                .commit_from_file(path.as_ref())?;
            Ok(session)
        }
        #[cfg(feature = "directml")]
        ExecutionProvider::DirectML => {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .with_intra_threads(4)?
                .commit_from_file(path.as_ref())?;
            Ok(session)
        }
        #[cfg(feature = "openvino")]
        ExecutionProvider::OpenVINO => {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([OpenVINOExecutionProvider::default().build()])?
                .with_intra_threads(4)?
                .commit_from_file(path.as_ref())?;
            Ok(session)
        }
    }
}
