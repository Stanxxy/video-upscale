"""Dump per-stage timing for an M0 job from Keyspaces checkpoints + lifecycle.

Produces:
  - <out_dir>/checkpoints.json  - raw checkpoints list
  - <out_dir>/lifecycle.json    - final lifecycle row
  - <out_dir>/timing.json       - machine-readable per-stage wall time
  - <out_dir>/timing_table.md   - human-readable markdown
  - <out_dir>/analysis_final.json (if uploaded)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# Stage order is fixed (V1 envelope ordering)
STAGE_ORDER = ["download", "detect", "track", "upscale_analyze", "annotate", "upload", "publish"]


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        if isinstance(s, datetime):
            return s.timestamp()
        return datetime.fromisoformat(str(s)).timestamp()
    except Exception:
        return None


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wall-start", type=float, required=True)
    ap.add_argument("--wall-end", type=float, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    load_dotenv()

    from service.jobs_store import JobsStore
    from service.keyspaces_client import KeyspacesClient

    client = KeyspacesClient()
    store = JobsStore(client)

    lifecycle = await store.get_lifecycle(args.job_id)
    if lifecycle:
        with open(os.path.join(args.out_dir, "lifecycle.json"), "w") as f:
            json.dump(lifecycle, f, indent=2, default=str)

    checkpoints = await store.get_all_checkpoints(args.job_id)
    with open(os.path.join(args.out_dir, "checkpoints.json"), "w") as f:
        json.dump(checkpoints, f, indent=2, default=str)

    # Per-stage timing: for each stage, take the earliest checkpoint and latest checkpoint.
    # Stage wall time = latest - earliest within a stage. Between-stage gaps are computed
    # using the latest of stage N vs. earliest of stage N+1.
    by_stage: dict[str, list[dict]] = {}
    for cp in checkpoints:
        by_stage.setdefault(cp["stage_name"], []).append(cp)

    for stage, rows in by_stage.items():
        rows.sort(key=lambda r: r.get("updated_at") or "")

    stage_summaries: list[dict] = []
    for stage in STAGE_ORDER:
        rows = by_stage.get(stage, [])
        if not rows:
            stage_summaries.append({"stage": stage, "checkpoint_count": 0})
            continue
        first = rows[0]
        last = rows[-1]
        first_ts = _iso_to_epoch(first["updated_at"])
        last_ts = _iso_to_epoch(last["updated_at"])
        stage_summaries.append({
            "stage": stage,
            "checkpoint_count": len(rows),
            "first_updated_at": str(first["updated_at"]),
            "last_updated_at": str(last["updated_at"]),
            "first_reason": (first.get("checkpoint_data") or {}).get("reason"),
            "last_reason": (last.get("checkpoint_data") or {}).get("reason"),
            "stage_wall_sec": (last_ts - first_ts) if (first_ts and last_ts) else None,
            "first_epoch": first_ts,
            "last_epoch": last_ts,
        })

    # Compute true per-stage wall time as "from end of prior stage (or wall_start) to end of this stage".
    prev_end = args.wall_start
    for s in stage_summaries:
        if s.get("last_epoch"):
            s["stage_wall_from_prev_sec"] = s["last_epoch"] - prev_end
            prev_end = s["last_epoch"]
        else:
            s["stage_wall_from_prev_sec"] = None

    total_wall = args.wall_end - args.wall_start
    pct_total = total_wall if total_wall > 0 else 1.0

    timing = {
        "job_id": args.job_id,
        "wall_start_epoch": args.wall_start,
        "wall_end_epoch": args.wall_end,
        "total_wall_sec": total_wall,
        "stages": stage_summaries,
        "lifecycle_final_state": lifecycle.get("job_state") if lifecycle else None,
        "lifecycle_error": lifecycle.get("error_message") if lifecycle else None,
    }
    with open(os.path.join(args.out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2, default=str)

    # Markdown table
    lines = ["# M0 timing table", "", f"job_id: `{args.job_id}`",
             f"total wall (driver-observed): **{total_wall:.1f} s**",
             f"lifecycle terminal state: **{timing['lifecycle_final_state']}**",
             ""]
    lines.append("| stage | checkpoints | first_reason | last_reason | stage_wall_intra (s) | stage_wall_from_prev (s) | % of total |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for s in stage_summaries:
        intra = s.get("stage_wall_sec")
        fromprev = s.get("stage_wall_from_prev_sec")
        pct = (fromprev / pct_total * 100) if fromprev else 0
        lines.append("| {stage} | {n} | {fr} | {lr} | {intra} | {fromprev} | {pct:.1f}% |".format(
            stage=s["stage"], n=s.get("checkpoint_count", 0),
            fr=s.get("first_reason", ""), lr=s.get("last_reason", ""),
            intra=f"{intra:.1f}" if intra is not None else "-",
            fromprev=f"{fromprev:.1f}" if fromprev is not None else "-",
            pct=pct,
        ))
    with open(os.path.join(args.out_dir, "timing_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))

    # Try to download analysis_final.json from the lifecycle's output bucket
    if lifecycle:
        # Look at the upload checkpoint artifacts for the analysis key
        analysis_key = None
        output_bucket = None
        for cp in checkpoints:
            arts = (cp.get("checkpoint_data") or {}).get("artifacts") or {}
            if arts.get("analysis_s3_key"):
                analysis_key = arts["analysis_s3_key"]
            if arts.get("analysis_bucket"):
                output_bucket = arts["analysis_bucket"]
        if analysis_key:
            # Try the request first for the output bucket
            try:
                rq = await store.get_request(args.job_id)
                if rq:
                    rqd = json.loads(rq)
                    output_bucket = rqd.get("output_bucket") or rqd.get("bucket") or output_bucket
            except Exception:
                pass
            print(f"[m0] analysis_s3_key={analysis_key} bucket={output_bucket}")
            if output_bucket:
                try:
                    from service.s3 import S3Client
                    s3 = S3Client(
                        region=os.environ.get("BJJ_AWS_REGION", "us-east-1"),
                        endpoint_url=os.environ.get("BJJ_S3_ENDPOINT_URL") or None,
                        access_key_id=os.environ.get("BJJ_AWS_ACCESS_KEY_ID") or None,
                        secret_access_key=os.environ.get("BJJ_AWS_SECRET_ACCESS_KEY") or None,
                    )
                    local = os.path.join(args.out_dir, "analysis_final.json")
                    s3.download_file(output_bucket, analysis_key, local)
                    print(f"[m0] downloaded analysis_final.json -> {local}")
                except Exception as e:
                    print(f"[m0] WARN: could not download analysis json: {e}")

    client.close()


if __name__ == "__main__":
    asyncio.run(amain())
