"""GPU/model cleanup and partial tracking JSON loader."""
import json
import re

def _ensure_models_released():
    """Release all ML models from GPU/RAM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ``torch.mps`` exists on Apple Silicon builds even when the runtime
        # has no MPS device (e.g. sandboxed CI); calling ``empty_cache()`` in
        # that state segfaults the process. Only release when MPS is usable.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Partial-tracking JSON loader (tolerates truncated streaming output)
# ---------------------------------------------------------------------------

def _load_partial_tracking_dict(partial_path: str) -> dict:
    """Parse tracking.json; tolerate truncated streaming output from abrupt cancellation."""
    with open(partial_path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r",\s*\"frames\"\s*:\s*\[", content)
    if not m:
        raise json.JSONDecodeError("no frames array in partial tracking", content, 0)
    header_raw = content[: m.start()] + "}"
    header = json.loads(header_raw)
    tail = content[m.end() :]
    dec = json.JSONDecoder()
    frames = []
    pos = 0
    n = len(tail)
    while pos < n:
        while pos < n and tail[pos] in " \t\n\r,":
            pos += 1
        if pos >= n:
            break
        try:
            obj, end = dec.raw_decode(tail, pos)
            frames.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return {**header, "frames": frames}
