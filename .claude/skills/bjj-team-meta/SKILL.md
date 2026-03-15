# Meta - Knowledge Capture

## Trigger
End of substantial conversations, significant architecture decisions, technical insights discovered during debugging, requirement changes or clarifications.

## Core Workflow

1. **Reflect** on what was accomplished, decided, or learned
2. **Categorize** as one of:
   - `decision` — Architecture, technology, or product choices with rationale
   - `insight` — Technical learnings, gotchas, debugging discoveries
   - `requirement` — Captured or changed requirements with acceptance criteria
3. **Save** to appropriate directory under `working_log/knowledge-base/`
4. **Update** `working_log/knowledge-base/INDEX.md` with new entry

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

## When NOT to Trigger
- Trivial changes (typo fixes, formatting)
- Changes already documented elsewhere
- Session-specific debugging steps that won't recur
