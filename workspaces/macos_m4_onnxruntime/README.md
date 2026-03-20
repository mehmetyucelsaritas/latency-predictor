# Mac M4 (PyTorch -> ONNX Runtime) Workspace

This workspace is pre-configured for nn-Meter **builder** to profile latency locally on Apple Silicon (M4-class CPU) using:

- Kernel/testcase generation: `IMPLEMENT: torch` (exports `.onnx`)
- Profiling backend: `mac_m4_onnxruntime` (runs ONNX with `onnxruntime` on CPU)

Workspace path:
`workspaces/macos_m4_onnxruntime`

## Suggested next steps

1. Install dependencies on your machine (you need `torch` + `onnx` + `onnxruntime`).
2. In Python, call `builder_config.init(workspace_path=...)`.
3. Use the backend name `mac_m4_onnxruntime` with `connect_backend()`.

