---
date: 2026-03-15
category: decision
tags: [service, api, taxonomy, frontend, gemini]
status: active
---

# Taxonomy Mapper for Frontend Enum Bridging

## Context
The Gemini analysis pipeline produces freeform technique descriptions. The bjj-vision
frontend expects strict enum values (e.g. `GUARD_PLAY`, `SUBMISSION_OFFENSE`). These
were mismatched, causing frontend parse errors.

## Content
`service/taxonomy_mapper.py` translates raw Gemini output strings into the frontend-compatible
enum values defined in `bjj_analysis_taxonomy.md`. It is applied as a post-processing step
inside `service/worker.py` before results are uploaded to S3 / published to SNS.

The Gemini prompt was also updated (commit aa3125c → c143cda) to output values closer
to the target taxonomy, reducing mapper complexity.

## Rationale
Keeping mapping logic in a dedicated module (rather than inside the prompt or in the
frontend) makes it testable and evolvable without re-deploying frontend or changing
the Gemini prompt contract.

## Impact
- `service/taxonomy_mapper.py` — the mapper
- `service/worker.py` — applies mapper after Gemini analysis
- `tests/test_taxonomy_mapper.py` — unit tests
- `bjj_analysis_taxonomy.md` — source of truth for valid enum values
