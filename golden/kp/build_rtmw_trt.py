"""Build a TensorRT FP16 engine for RTMW-x (133 COCO-WholeBody keypoints) from the OpenMMLab ONNX export.
Usage: build_rtmw_trt.py ONNX ENGINE  (batch 1-4, 384x288 input)"""
import sys
import tensorrt as trt

onnx_path, engine_path = sys.argv[1:3]
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
prof.set_shape("input", (1, 3, 384, 288), (2, 3, 384, 288), (4, 3, 384, 288))
cfg.add_optimization_profile(prof)
plan = builder.build_serialized_network(net, cfg)
if plan is None:
    raise SystemExit("engine build failed")
open(engine_path, "wb").write(plan)
print("built", engine_path, plan.nbytes // 1e6, "MB")
