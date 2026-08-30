"""Bezos precondition 1, second half (2026-07-26 authorize-with-preconditions
review): run `RerouteUploadCache.ensure()` for `w37` (Gate 1) and `occ11`
(Gate 2) — download+normalize+upload+poll ACTIVE — with ZERO Gemini
`generateContent` spend (no position/technique/validator calls at all).
De-risks the two §G.1-rerouted windows that were not touched by the earlier
reroute smoke test (which only exercised `occ0`), and records
`upload_latency_ms`/`active_wait_ms` to `gate_budget_ledger.json` for both.

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.derisk_reroute_uploads
"""
from __future__ import annotations

import asyncio
import sys

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc

TARGETS = [
    ("gate1", gc.GATE1_MANIFEST_PATH, "windows", "66b009481c62a1db7a53d504__w37"),
    ("gate2", gc.GATE2_MANIFEST_PATH, "clips", "66a6ff021c62a1db7a5133c0__occ11"),
]


async def _amain() -> None:
    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    ledger = gc.CallLedger()

    for gate_tag, manifest_path, key, window_id in TARGETS:
        manifest = gc.read_json(manifest_path)
        row = next((r for r in manifest[key] if r["window_id"] == window_id), None)
        if row is None:
            raise RuntimeError(f"derisk_reroute_uploads: {window_id!r} not found in {manifest_path}")
        assert window_id in reroute.REROUTE_WINDOW_IDS, "target must be a §G.1 rerouted window"

        upload_cache = reroute.RerouteUploadCache(client, ledger, gate=gate_tag)
        print(f"[derisk] {gate_tag} {window_id} — ensuring upload (download+normalize+upload+poll, "
              "ZERO generateContent calls)...", file=sys.stderr)
        upload = await upload_cache.ensure(row)
        print(f"[derisk] {window_id}: file_uri={upload.file_uri}", file=sys.stderr)
        print(f"  download_latency_ms={upload.download_latency_ms:.0f} "
              f"normalize_latency_ms={upload.normalize_latency_ms:.0f} "
              f"UPLOAD_LATENCY_MS={upload.upload_latency_ms:.0f} "
              f"ACTIVE_WAIT_MS={upload.active_wait_ms:.0f} sec_start_s={upload.sec_start_s}", file=sys.stderr)
        ledger.flush()

    print("[derisk] DONE. 0 Gemini generateContent calls made. See gate_budget_ledger.json for "
          "the files_api_upload rows (w37/occ11) and reroute_manifest.json for full provenance.",
          file=sys.stderr)


def main(argv=None) -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
