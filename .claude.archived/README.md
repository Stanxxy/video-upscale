# Archived `.claude` (2026-05-25)

This directory is a frozen copy of the repo-local Claude Code config that lived at `whole-video-analysis/.claude/`.

## Why archived

The six `bjj-team-*` skills here were a **stale fork** of the canonical suite at the **umbrella workspace**:

```
bjj-proj/.claude/skills/
```

The umbrella skills are newer and aligned with the post-split layout (nested git repos, Docker LocalStack integration harness, evaluator loop protocol, etc.). Keeping a second copy in this repo caused drift (e.g. deprecated VPN LocalStack `100.79.167.101:4566`, outdated monorepo assumptions).

## Canonical replacement

Use skills from `bjj-proj/.claude/skills/` (see `whole-video-analysis/AGENTS.md` → Team Skills).

Vision-engine-specific **hygiene scan paths** from the old `bjj-team-operator` skill were merged into the umbrella operator skill (section **Vision engine artifacts**).

## Preserved local settings (not in git)

If you relied on repo-local permissions, migrate manually:

- **`settings.json`**: allowed `ssh gx10`, `scp`, `rsync` for remote GPU host access.
- **`settings.local.json`**: machine-specific dev permissions (venv paths, etc.) — was never tracked in git.

Do not restore this tree unless you intentionally want to fork skills again.
