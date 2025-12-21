# vad-rs

Speech detection using Silero VAD in Rust with GPU acceleration support.

## Install

```toml
[dependencies]
vad-rs = "0.1.5"
```

## GPU Acceleration

Enable GPU acceleration via Cargo features:

| Feature | GPU Type | Platform | Requirements |
|---------|----------|----------|--------------|
| `coreml` | Apple GPU/Neural Engine | macOS/iOS | Xcode |
| `cuda` | NVIDIA GPU | Linux/Windows | CUDA 11.6+ |
| `tensorrt` | NVIDIA GPU (optimized) | Linux/Windows | TensorRT |
| `directml` | AMD/Intel/NVIDIA | Windows | DirectX 12 |
| `openvino` | Intel CPU/GPU/NPU | Linux/Windows | OpenVINO |

Example with CoreML (macOS):

```toml
[dependencies]
vad-rs = { version = "0.1.5", features = ["coreml"] }
```

## Examples

See [examples](examples)

```bash
# Download model
wget https://github.com/thewh1teagle/vad-rs/releases/download/v0.1.0/silero_vad.onnx

# Run with GPU (macOS)
cargo run --example segment --features coreml -- silero_vad.onnx audio.wav
```
