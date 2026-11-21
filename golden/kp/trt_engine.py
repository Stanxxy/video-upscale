"""TensorRT engine wrapper shared by the pose and detector stages (one input named "input", float32 outputs).

Outputs whose size depends on the data (RTMO's detection count) are received through an output allocator."""
import numpy as np
import tensorrt as trt
import torch


class _Alloc(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.buf, self.shape = None, None

    def reallocate_output(self, tensor_name, memory, size, alignment):
        self.buf = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
        return self.buf.data_ptr()

    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
        return self.reallocate_output(tensor_name, memory, size, alignment)

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(shape)


class Engine:
    def __init__(self, path):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.allocs = {n: _Alloc() for n in self.outs}
        for n, a in self.allocs.items():
            self.ctx.set_output_allocator(n, a)

    def __call__(self, batch):   # batch: float32 [N, 3, H, W]; returns the outputs in engine order
        x = torch.from_numpy(batch).cuda().contiguous()
        self.ctx.set_input_shape("input", tuple(x.shape))
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        out = []
        for n in self.outs:
            a = self.allocs[n]
            count = int(np.prod(a.shape))
            out.append(a.buf[:count * 4].view(torch.float32).reshape(a.shape).cpu().numpy())
        return out
