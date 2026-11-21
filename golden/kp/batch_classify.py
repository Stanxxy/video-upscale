"""The classifier through Gemini's Batch API (half the standard price; results within 24 hours).

Builds the same requests as classify.py (raw frames), one per window and repeat, as a JSONL file, uploads it with the
Files API and creates one batch job. `collect` polls the job and, when it has succeeded, writes rows in classify.py's
format (window, cond, rep, answer, tokens, cost at the batch price, model) so score.py and f1gold.py read them as
they are. Turnaround (create to done) is recorded in the job file.

Usage:
  batch_classify.py submit WINDOWS_JSON PACK_DIR JOB_JSON --names A B [--repeat N] [--model M] [--thinking T] [--res R]
  batch_classify.py collect JOB_JSON OUT_JSONL          (exit 0 when written, 3 while the job is still running)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

import classify

API = "https://generativelanguage.googleapis.com"
BATCH_SHARE = 0.5          # batch requests are billed at half the standard price


def body(parts, thinking, res):
    return {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": classify.SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": thinking},
                                 "mediaResolution": f"MEDIA_RESOLUTION_{res}"}}


def upload(path, key):
    size = path.stat().st_size
    r = httpx.post(f"{API}/upload/v1beta/files", headers={
        "x-goog-api-key": key, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "application/jsonl",
        "Content-Type": "application/json"}, json={"file": {"display_name": path.name}}, timeout=60)
    r.raise_for_status()
    url = r.headers["x-goog-upload-url"]
    r = httpx.post(url, headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                 "X-Goog-Upload-Command": "upload, finalize"}, content=path.read_bytes(), timeout=900)
    r.raise_for_status()
    return r.json()["file"]["name"]


def submit(a):
    classify.PACK = Path(a.pack)
    wins = json.load(open(a.windows))
    index = json.load(open(Path(a.pack) / f"index_{Path(a.windows).stem}.json"))
    key = classify.key()
    job = Path(a.job)
    lines = job.with_suffix(".requests.jsonl")
    with open(lines, "w") as fh:
        for rep in range(a.repeat):
            for w in wins:
                parts = classify.build(w, index[w["id"]], "raw", a.names, [])
                fh.write(json.dumps({"key": f"{w['id']}|{rep}", "request": body(parts, a.thinking, a.res)}) + "\n")
    name = upload(lines, key)
    r = httpx.post(f"{API}/v1beta/models/{a.model}:batchGenerateContent", headers={"x-goog-api-key": key},
                   json={"batch": {"display_name": job.stem, "input_config": {"file_name": name}}}, timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code} {r.text[:400]}")
    batch = r.json()
    json.dump(dict(batch=batch["name"], model=a.model, thinking=a.thinking, res=a.res, repeat=a.repeat,
                   windows=len(wins), requests=len(wins) * a.repeat, input_file=name,
                   input_mb=round(lines.stat().st_size / 1e6, 1), created=time.time()), open(job, "w"), indent=1)
    lines.unlink()
    print("submitted", batch["name"], len(wins) * a.repeat, "requests")


def collect(job_path, out):
    job = json.load(open(job_path))
    key = classify.key()
    r = httpx.get(f"{API}/v1beta/{job['batch']}", headers={"x-goog-api-key": key}, timeout=60)
    r.raise_for_status()
    b = r.json()
    meta = b.get("metadata", b)
    state = meta.get("state") or b.get("state")
    if state not in ("JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"):
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "BATCH_STATE_FAILED",
                     "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"):
            raise SystemExit(f"{job['batch']} ended {state}: {json.dumps(b)[:400]}")
        print(job["batch"], state, f"{(time.time() - job['created']) / 60:.0f} min")
        sys.exit(3)
    resp = b.get("response") or meta.get("output") or {}
    fname = resp.get("responsesFile") or meta.get("dest", {}).get("fileName") or b.get("dest", {}).get("fileName")
    if not fname:
        raise SystemExit(f"no responses file in {json.dumps(b)[:600]}")
    data = httpx.get(f"{API}/download/v1beta/{fname}:download", params={"alt": "media"},
                     headers={"x-goog-api-key": key}, timeout=600).text
    price_in, price_out = classify.PRICES[job["model"]]
    rows, failed = [], []
    for line in data.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        k = item.get("key") or item.get("metadata", {}).get("key")
        if "error" in item or "response" not in item:
            failed.append(dict(key=k, error=item.get("error")))
            continue
        v = item["response"]
        u = v.get("usageMetadata", {})
        tin, tout = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        window, rep = k.rsplit("|", 1)
        rows.append(dict(window=window, cond="raw", rep=int(rep),
                         answer=json.loads(v["candidates"][0]["content"]["parts"][-1]["text"]),
                         tokens_in=tin, tokens_out=tout,
                         cost=round((tin * price_in / 1e6 + tout * price_out / 1e6) * BATCH_SHARE, 5),
                         model_version=v.get("modelVersion"), batch=True))
    with open(out, "w") as fh:
        for row in sorted(rows, key=lambda r: (r["rep"], r["window"])):
            fh.write(json.dumps(row) + "\n")
    job.update(done=time.time(), turnaround_min=round((time.time() - job["created"]) / 60, 1), ok=len(rows),
               failed=len(failed), failures=failed[:5], cost=round(sum(r["cost"] for r in rows), 3))
    json.dump(job, open(job_path, "w"), indent=1)
    print(f"{job['batch']}: {len(rows)} ok, {len(failed)} failed, ${job['cost']}, {job['turnaround_min']} min")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("windows"); ap.add_argument("pack"); ap.add_argument("job")
        ap.add_argument("--names", nargs=2, required=True); ap.add_argument("--repeat", type=int, default=1)
        ap.add_argument("--model", default="gemini-4-hydrogen", choices=sorted(classify.PRICES))
        ap.add_argument("--thinking", default="HIGH"); ap.add_argument("--res", default="HIGH")
        submit(ap.parse_args())
