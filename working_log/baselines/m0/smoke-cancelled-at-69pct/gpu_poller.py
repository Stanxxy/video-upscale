import csv, time, sys, os
import subprocess
def smi_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        return out.split("\n")[0]
    except Exception:
        return ""
def torch_mem():
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1024 / 1024
            r = torch.cuda.memory_reserved() / 1024 / 1024
            return f"{a:.1f}", f"{r:.1f}"
    except Exception:
        pass
    return "", ""
def sys_mem():
    try:
        out = subprocess.check_output(["free","-m"], text=True, timeout=5).splitlines()
        # Mem: line: total used free shared buff/cache available
        parts = out[1].split()
        return parts[2], parts[1]
    except Exception:
        return "", ""
INTERVAL = float(os.environ.get("GPU_SAMPLE_INTERVAL","30"))
OUT = sys.argv[1]
while True:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    u = smi_util()
    a, r = torch_mem()
    used, total = sys_mem()
    with open(OUT, "a") as f:
        f.write(f"{ts},{u},{a},{r},{used},{total}\n")
    time.sleep(INTERVAL)
