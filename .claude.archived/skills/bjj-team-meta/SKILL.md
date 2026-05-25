---
name: bjj-team-meta
description: Captures durable project decisions, insights, and requirement changes into the knowledge base. Use when wrapping up meaningful work or documenting important learnings.
when_to_use: Trigger for every substantial completed task to capture decisions, insights, requirements, and keep INDEX.md aligned with verified evidence.
paths:
  - working_log/knowledge-base/**
  - working_log/**
  - whole-video-analysis/working_log/**
  - whole-video-analysis/working_log/knowledge-base/**
  - .claude/skills/**
  - .claude/agents/**
user-invocable: false
---

# Meta - Knowledge Capture

## Trigger
End of substantial conversations, significant architecture decisions, technical insights discovered during debugging, requirement changes or clarifications.

## Knowledge Base Preflight (Always-On)
Before any capture or maintenance action:

1. Read `working_log/knowledge-base/INDEX.md`.
2. Query related KB entries in the target category to avoid duplicates and conflicts.
3. Compare candidate updates against current code reality (files, symbols, tests, logs, and accepted plans).
4. If evidence is insufficient, request missing proof before finalizing lifecycle changes.

## Core Workflow

1. **Reflect** on what was accomplished, decided, or learned
2. **Categorize** as one of:
   - `decision` — Architecture, technology, or product choices with rationale
   - `insight` — Technical learnings, gotchas, debugging discoveries
   - `requirement` — Captured or changed requirements with acceptance criteria
3. **Save** to appropriate directory under `working_log/knowledge-base/`
4. **Update** `working_log/knowledge-base/INDEX.md` with new entry

## Governance Decision Evidence (Mandatory for Architecture-Impacting Work)
When a task is marked architecture-impacting, Meta must verify governance package evidence before closing capture:

1. **AIP evidence**
   - global architecture fit across services, API contracts, data flow, and infrastructure/runtime behavior
   - explicit invariants/constraints and rejected alternatives
2. **Execution task graph evidence**
   - ordered implementation phases with dependency links
3. **Migration and operations runbook evidence**
   - rollout checks, observability checks, rollback steps, and rollback verification
4. **Enforcement evidence**
   - milestone recheck status (`M1`, `M2`, `M3`)
   - deviation re-review records (if any)
   - owner override audit note (if override used)

If any artifact is missing, mark completion gate as failed and request remediation.

## Lifecycle Management Authority (Safe-Auto)
Meta owns KB document lifecycle for `working_log/knowledge-base/`.

### Safe-Auto Allowed (no extra approval)
- Update entry metadata (`status`, `tags`, impacted files/services) when evidence is clear.
- Refresh content sections (`Context`, `Content`, `Rationale`, `Impact`) to match validated code changes.
- Mark outdated claims as corrected when replacement evidence is present in current code/tests.

### Approval Required (must ask user first)
- Deleting KB files.
- Merging multiple KB entries.
- Marking an entry `superseded` when replacement reference is uncertain.
- Any destructive bulk cleanup action.

## Strict Alignment Checks (KB Must Match Code)
Before finalizing updates, verify each claim against real codebase evidence:
- **Path check**: referenced files/directories still exist and are relevant.
- **Symbol check**: referenced APIs/functions/services are present or intentionally retired.
- **Behavior check**: tests, build output, or runtime evidence support the documented behavior.
- **Conflict check**: no active KB entry contradicts the new claim without explicit supersession.

If alignment fails, do not finalize silently: create a correction proposal with evidence.

## Proposal-First Obsolescence Workflow
For stale or obsolete knowledge docs, Meta must first prepare a proposal report:

1. List candidate entries with staleness/conflict reason.
2. Attach conflicting code evidence (paths/symbols/tests/logs).
3. Recommend one action per entry: `update`, `supersede`, or `delete`.
4. Ask for approval before any delete/merge/supersede action.

## Knowledge Base Entry Format

```markdown
---
date: YYYY-MM-DD
category: decision | insight | requirement
tags: [relevant, tags]
status: active | superseded | deprecated
---

# Title

## Context
Why this came up.

## Content
The decision, insight, or requirement itself.

## Rationale
Why this choice was made (for decisions) or why this matters (for insights).

## Impact
What this affects — files, services, workflows.
```

## File Naming
`{category}/{YYYY-MM-DD}-{kebab-case-title}.md`

Example: `decisions/2026-03-07-use-zustand-for-video-state.md`

## Separation Rules
- **Project decisions and learnings** go in `working_log/knowledge-base/`
- **Environment setup facts** (tool versions, venv paths, macOS quirks) go in `~/.claude/projects/.../memory/MEMORY.md`
- **Never duplicate** between the two — pick the right home

## INDEX.md Maintenance
When adding an entry, append it under the correct section:
```markdown
## Decisions
- [2026-03-07 - Use Zustand for Video State](decisions/2026-03-07-use-zustand-for-video-state.md) — tags: frontend, state
```

For governance-driven architecture decisions, include `governance` and at least one of `microservice`, `api-contract`, `data-model`, `infra` in tags.
Prefer summaries that explicitly state the architecture invariant being enforced.

## When NOT to Trigger
- Trivial changes (typo fixes, formatting)
- Changes already documented elsewhere
- Session-specific debugging steps that won't recur

## Required Output Contract (Mandatory)
Every Meta response must include these sections:

1. `## Capture Decision`
2. `## Evidence Links`
3. `## KB Updates Applied`
4. `## Completion Gate Status`

`Completion Gate Status` must explicitly state whether mandatory capture requirements are satisfied.
For architecture-impacting tasks, `Evidence Links` must reference AIP, task graph, runbook, and milestone evidence.
