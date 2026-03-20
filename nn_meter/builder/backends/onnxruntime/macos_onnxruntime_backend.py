import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..interface import BaseBackend, BaseParser, BaseProfiler
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults


class MacM4ONNXRuntimeLatencyParser(BaseParser):
    """Parse ONNXRuntime profiling output into nn-Meter Latency."""

    def __init__(self):
        self.total_latency = Latency()

    def parse(self, content: Any):
        if isinstance(content, dict):
            data = content
        else:
            data = json.loads(content)

        self.total_latency = Latency(
            avg=float(data["latency_avg_ms"]),
            std=float(data["latency_std_ms"]),
        )
        return self

    @property
    def latency(self) -> Latency:
        return self.total_latency

    @property
    def results(self):
        return ProfiledResults({"latency": self.latency})


class MacM4ONNXRuntimeProfiler(BaseProfiler):
    """Local CPU profiler for ONNX models using ONNX Runtime."""

    def __init__(
        self,
        num_runs: int = 50,
        warm_ups: int = 10,
        providers: Optional[List[str]] = None,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
        graph_optimization_level: Optional[str] = None,
        batch_size: int = 1,
        seed: int = 0,
    ):
        self.num_runs = int(num_runs)
        self.warm_ups = int(warm_ups)
        self.providers = providers
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.graph_optimization_level = graph_optimization_level
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        # Keep deterministic-ish input generation overhead out of the timed section.
        np.random.seed(self.seed)

    def _normalize_providers(self):
        if self.providers is None:
            return ["CPUExecutionProvider"]
        if isinstance(self.providers, str):
            return [self.providers]
        return list(self.providers)

    def _resolve_providers(self, ort):
        requested = self._normalize_providers()
        available = set(ort.get_available_providers())

        # Keep requested priority order, but only use providers present in this ORT build.
        selected = [provider for provider in requested if provider in available]
        if selected:
            return selected

        # Hard fallback for this backend: always use CPU EP when CoreML is unavailable.
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]

        # Last resort for unusual ORT builds that do not expose CPU EP.
        available_list = ort.get_available_providers()
        if available_list:
            return [available_list[0]]
        raise RuntimeError("No ONNX Runtime execution providers are available.")

    def _normalize_input_shapes(self, input_shape, session_inputs):
        """
        nn-Meter passes `input_shape` as a list of tensor shapes (no batch dim).
        For typical cases:
          - single input: [[CIN, H, W]]
          - multi input:  [[...], [...]]
        Some callers might pass a single shape list like [CIN] or [CIN, H, W].
        """
        if input_shape is None:
            shapes = []
            for inp in session_inputs:
                s = list(inp.shape)
                # s looks like [N, C, H, W] where N may be symbolic.
                dims = []
                for d in s[1:]:
                    dims.append(int(d) if isinstance(d, int) and d > 0 else 1)
                shapes.append(dims)
            return shapes

        # If it's a flat list of ints (e.g., [CIN, H, W]), treat it as a single input.
        if isinstance(input_shape, list) and input_shape and isinstance(input_shape[0], int):
            return [list(input_shape)]

        # Otherwise assume it's already a list-of-shapes.
        return input_shape

    def profile(self, model_path: str, input_shape=None, **kwargs):
        import onnxruntime as ort

        providers = self._resolve_providers(ort)
        sess_options = ort.SessionOptions()

        if self.intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = int(self.intra_op_num_threads)
        if self.inter_op_num_threads is not None:
            sess_options.inter_op_num_threads = int(self.inter_op_num_threads)

        if self.graph_optimization_level:
            level = str(self.graph_optimization_level).lower()
            if level in ("all", "enable_all", "ort_enable_all"):
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            elif level in ("basic", "enable_basic", "ort_enable_basic"):
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif level in ("none", "disable", "ort_disable"):
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        session_inputs = session.get_inputs()
        input_names = [i.name for i in session_inputs]

        # Determine batch size from exported model input, falling back to config.
        batch_dim = session_inputs[0].shape[0] if session_inputs else None
        batch_size = (
            int(batch_dim)
            if isinstance(batch_dim, int) and batch_dim > 0
            else self.batch_size
        )

        shapes = self._normalize_input_shapes(input_shape, session_inputs)

        # Pre-generate random inputs so the timer only measures model execution.
        input_feed: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(input_names):
            spec = shapes[idx] if idx < len(shapes) else shapes[0]
            spec = list(spec)
            input_feed[name] = np.random.rand(batch_size, *spec).astype(np.float32)

        latencies_ms: List[float] = []
        total_iters = self.warm_ups + self.num_runs
        for i in range(total_iters):
            start = time.perf_counter()
            session.run(None, input_feed)
            end = time.perf_counter()
            if i >= self.warm_ups:
                latencies_ms.append((end - start) * 1000.0)

        avg = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        std = float(np.std(latencies_ms)) if latencies_ms else 0.0
        return json.dumps(
            {
                "latency_avg_ms": avg,
                "latency_std_ms": std,
                "num_runs": self.num_runs,
                "warm_ups": self.warm_ups,
            }
        )


class MacM4ONNXRuntimeBackend(BaseBackend):
    parser_class = MacM4ONNXRuntimeLatencyParser
    profiler_class = MacM4ONNXRuntimeProfiler

    def update_configs(self):
        super().update_configs()
        cfg = self.configs or {}

        self.profiler_kwargs.update(
            {
                "num_runs": cfg.get("NUM_RUNS", 50),
                "warm_ups": cfg.get("WARM_UPS", 10),
                "providers": cfg.get("PROVIDERS", ["CPUExecutionProvider"]),
                "intra_op_num_threads": cfg.get("INTRA_OP_NUM_THREADS", None),
                "inter_op_num_threads": cfg.get("INTER_OP_NUM_THREADS", None),
                "graph_optimization_level": cfg.get("GRAPH_OPT_LEVEL", None),
                "batch_size": cfg.get("BATCH_SIZE", 1),
                "seed": cfg.get("SEED", 0),
            }
        )

    def convert_model(self, model_path, save_path, input_shape=None):
        # nn-Meter's torch kernel/testcase generators already export ONNX files.
        # We keep the conversion step as a pass-through for `.onnx` models.
        if os.path.isdir(model_path):
            candidates = [
                f
                for f in os.listdir(model_path)
                if f.endswith(".onnx") and os.path.isfile(os.path.join(model_path, f))
            ]
            if not candidates:
                raise FileNotFoundError(f"No .onnx file found under: {model_path}")
            if len(candidates) > 1:
                raise ValueError(f"Expected a single .onnx, found: {candidates}")
            return os.path.join(model_path, candidates[0])

        if os.path.isfile(model_path) and model_path.endswith(".onnx"):
            return model_path

        # Some callers may pass a path without extension (e.g. `foo` instead of `foo.onnx`).
        if os.path.isfile(model_path + ".onnx"):
            return model_path + ".onnx"

        raise ValueError(f"Unsupported model path for ONNXRuntime backend: {model_path}")

    def test_connection(self):
        # For local profiling there is nothing to "connect" to; just validate dependencies.
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required for this backend. Install it (e.g., `pip install onnxruntime`)."
            ) from e

