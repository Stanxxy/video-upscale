"""Build a TensorRT FP16 engine from an ONNX export (RTMW-x pose, RTMO person+pose detector).
Usage: build_trt.py ONNX ENGINE MIN OPT MAX     shapes as 1x3x384x288 (the input tensor is named "input")"""
import sys

import tensorrt as trt

onnx_path, engine_path = sys.argv[1:3]
shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[3:6]]
log = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(log)
net = builder.create_network(0)
parser = trt.OnnxParser(net, log)
if not parser.parse_from_file(onnx_path):
    raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
cfg = builder.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
prof = builder.create_optimization_profile()
prof.set_shape("input", *shapes)
cfg.add_optimization_profile(prof)
plan = builder.build_serialized_network(net, cfg)
if plan is None:
    raise SystemExit("engine build failed")
open(engine_path, "wb").write(plan)
print("built", engine_path, plan.nbytes // 1e6, "MB")
