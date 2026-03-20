from nn_meter.builder import builder_config, profile_models, build_latency_predictor
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases, detect_fusion_rule
builder_config.init("workspaces/macos_m4_onnxruntime")
backend = connect_backend("mac_m4_onnxruntime")
testcases = generate_testcases()
profiled = profile_models(backend, testcases, mode="ruletest")
detect_fusion_rule(profiled)
build_latency_predictor(backend="mac_m4_onnxruntime")