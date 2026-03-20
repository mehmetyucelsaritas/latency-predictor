"""
ONNX Runtime-based backend implementations.

This backend is intended for local CPU profiling on macOS (e.g., Apple Silicon).
"""

from .macos_onnxruntime_backend import MacM4ONNXRuntimeBackend

__all__ = ["MacM4ONNXRuntimeBackend"]

