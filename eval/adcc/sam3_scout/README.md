# SAM3 Scout (`eval/adcc/sam3_scout`)

Offline gx10 scout: Arms A/B/C vs ADCC gold windows. See
`working_log/plans/2026-07-20-sam3-scout-implementation.md`.

## Locks

- SAM3 **only on gx10** (concurrency 1, 30s chunks / 45s threshold)
- Gemini: **`gemini-3.5-flash` @ 10fps high** — never flash-lite
- No track-fidelity stop-gate (HTML eyeball + soft advisory)

## Quick path

```bash
# unit tests (local)
cd whole-video-analysis-wt-vlmstudio
python -m pytest eval/adcc/sam3_scout/tests/ -v

# cut clips where ADCC media root is mounted (often bjj-server), rsync to gx10
python -m eval.adcc.sam3_scout.cut_clips --rsync-to-gx10

# deploy code + dry-run on gx10
./eval/adcc/sam3_scout/deploy_sam3_scout_to_gx10.sh
ssh gx10 'bash /home/sliu1143/sam3_scout/code/sam3_scout/dry_run_one_clip.sh'
```

Serve `viewer/index.html` next to run artifacts; load a viewer JSON from
`viewer_manifest.write_viewer_bundle`.
