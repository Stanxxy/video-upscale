---
date: 2026-04-19
category: requirement
tags: [service, resume, lifecycle, keyspaces]
status: active
---

# Job Pause and Resume Open Questions

## Context
There is an existing note (`working_log/knowledge-base/decisions/job-pause-and-resume.md`) that captures unresolved pause/resume lifecycle questions, but it is not in the indexed KB format.

## Content
Open requirement questions retained as-is:

- `The current directory is search engine`
- `A job submit to search engine via endpoint /track`
- `Job lifecycle entry in keyspaces was initiated updated for the submitted job`
- `Save job request parameters in keyspaces (Still necessary)`
  - `Together with state pending?`

## Rationale
These statements/questions indicate pause/resume expectations are still being clarified. Keeping them as explicit requirements prevents accidental assumptions that lifecycle persistence semantics are settled.

## Impact
- Affects pause/resume behavior in `service/routes.py` and `service/worker.py`.
- Affects lifecycle/request persistence in `service/jobs_store.py`.
