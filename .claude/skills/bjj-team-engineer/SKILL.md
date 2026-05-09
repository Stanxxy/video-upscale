---
name: bjj-team-engineer
description: Provides implementation workflow standards for coding, refactoring, and bug fixing in BJJ Vision. Use when writing code changes, tests, and technical verification steps.
when_to_use: Trigger for implementation, bug fixes, refactors, or test-writing tasks that change behavior or production code.
paths:
  - bjj-vision-frontend/**
  - bjj-vision-backend/**
  - standalone-analyzer-backend/**
  - whole-video-analysis/**
  - scripts/**
  - working_log/**
  - whole-video-analysis/working_log/**
user-invocable: true
---

# Engineer - Implementation

## Trigger
Implementing features, fixing bugs, refactoring code, writing tests.

## Knowledge Base Preflight (Always-On)
Before implementation starts:

1. Read `working_log/knowledge-base/INDEX.md`.
2. Query relevant KB entries for impacted service, component, or subsystem.
3. Extract prior decisions, known failure modes, and validation expectations.
4. If no entry is relevant, continue and flag a potential KB gap for post-task capture.

## KB Update Triggers (Moderate Auto-Create)
Create or update a KB candidate when any of these are true:
- An accepted plan introduces architecture/API/data-model/infrastructure behavior changes.
- A user-visible behavior change is implemented and verified.
- A nontrivial bug fix is completed with root cause, fix details, and verification evidence.

Do not create KB entries for trivial edits (formatting, typo-only changes, no behavior effect).

When KB update criteria are met:
- Include concrete evidence (changed paths, key symbols, test/build output, logs).
- Hand off lifecycle ownership to `bjj-team-meta` for strict alignment checks and safe-auto updates.

## Governance Planning Stage (Mandatory for Architecture-Impacting Work)
Before TDD or implementation, run an architecture-impact triage:

- Service boundary changes (new responsibility split, ownership moves, cross-service coupling)
- API contract changes (REST/gRPC/WebSocket payloads, response shape, auth expectations)
- Data model/schema changes (tables, events, cache keys, migration semantics)
- Infrastructure/runtime/deployment changes (LocalStack/S3/DynamoDB/Cassandra/Supabase, startup dependencies, scaling/failure domain behavior)

If any item applies, mark the task as `architecture-impacting` and enforce the governance package below.

### Hard Governance Gate
- Engineering **must not** start implementation until governance planning artifacts are complete and approved.
- No code edits, no TDD cycle, and no migration work may begin before gate completion.

### Mandatory Governance Package (All Required)
1. **Architecture Implementation Plan (AIP)**
   - Global architecture fit across services, APIs, data flow, and infrastructure/runtime behavior.
   - Architectural invariants and constraints that implementation must preserve.
   - Tradeoff analysis with rejected alternatives.
2. **Execution Task Graph**
   - Ordered task breakdown with dependencies and sequence.
   - Contract-first order to prevent cross-service breakage.
3. **Migration & Operations Runbook**
   - Rollout and migration steps, backward compatibility, observability checks, and rollback plan.

### Enforcement Rules
- **Strict baseline**: Implementation must follow the approved governance package.
- **No silent drift**: Any architecture-relevant deviation requires governance re-review before proceeding.
- **Milestone rechecks** are mandatory for architecture-impacting work:
  - `M1`: contract/schema readiness
  - `M2`: mid-implementation integration status
  - `M3`: pre-QA conformance + rollout readiness

### Owner Override
- Override is allowed only with explicit owner approval.
- Override requires an audit note in KB including accepted risks, constrained scope, and mandatory post-change governance review.

## Core Workflow

1. **Branch** — Create a feature branch from `develop` before starting work: `git checkout develop && git pull && git checkout -b feature/<name>`. Never commit directly to `develop` or `main`.
2. **Understand** — Read relevant code before changing it. Never modify code you haven't read.
3. **Governance triage** — Determine whether work is architecture-impacting using the triage checklist.
4. **Governance package** — For architecture-impacting work, produce and approve AIP + task graph + runbook before coding.
5. **TDD cycle** — Write failing test, implement minimally, verify pass. Compose with `superpowers:test-driven-development` for structured TDD.
6. **Milestone rechecks** — For architecture-impacting work, complete M1/M2/M3 rechecks and re-review deviations.
7. **Build/lint check** — Run `npm run build` (frontend) or `pytest` (backend) to verify.
8. **Evaluator loop** — Hand off to Evaluator and iterate with Designer/Engineer feedback until all evaluator findings are closed with evidence. When the Evaluator lists **legacy code or comment removal** (or replacement) in `## Legacy and Dead-Code Findings` or in the **Engineer action list** under `## Next Handoff`, treat those as **mandatory implementation work**: delete dead paths, remove stale comments and commented-out blocks, update or delete obsolete docs in scope—do not leave them as “follow-up notes.” Re-run search and tests to prove the legacy surface is gone or correctly replaced.
9. **PM verification handoff** — After evaluator convergence, hand off to PM for final requirement-level verification before Meta capture.
10. **PM verification bounce-back** — When `bjj-team-product-manager` reports **`failed-returned-to-engineering`**, **`blocked-awaiting-engineering`**, or that the feature **cannot be verified** until behavior, harness, env, or docs change: re-enter **`bjj-team-evaluator`** then **`bjj-team-engineer`** until Evaluator converges again; PM re-runs verification. Do not treat work as complete toward Meta until PM `## PM Verification Loop Status` is **`passed`**. If PM is **`blocked-awaiting-user`**, support with docs, runnable URLs, or harnesses as soon as unblocked—do not require Meta until PM passes.

## Frontend Patterns
- **Feature modules**: `src/features/{feature-name}/`
- **Shared UI**: `src/components/ui/` (Radix + shadcn/ui primitives)
- **Types**: `src/types/` for shared TypeScript definitions
- **State**: React Context for auth/global, Zustand for complex feature state
- **Styling**: Tailwind CSS utility classes, no custom CSS unless unavoidable
- **Icons**: Lucide React exclusively

## Backend Patterns
- **Structure**: `app.py` → Controllers → Services → Repositories → Infrastructure
- **Async-first**: All I/O operations use `async/await`
- **Testing**: pytest with pytest-asyncio, fixtures in `conftest.py`
- **Service directory**: Each service has its own `run.py`, `src/app.py`, and `tests/`
- **Shared code**: `shared_lib/` for cross-service models and infrastructure

## Evaluator Collaboration Loop (Mandatory)
- The Engineer must enter a review loop with `bjj-team-evaluator` for behavior-impacting changes.
- Early exit is forbidden. Loop continues until all evaluator findings are resolved or explicitly waived with owner-approved rationale.
- Every rejection from evaluator must be addressed with concrete diffs and verification evidence.
- **Legacy and dead-code directives** from the Evaluator (including whole-repo findings outside the original diff) are in scope for the same loop: implement removal or replacement, cite paths touched, and supply verification evidence—same bar as functional bugs.
- Engineer may challenge evaluator findings, but closure requires explicit mutual agreement.

## Gates
- Architecture-impacting work has approved governance package (AIP + task graph + runbook)
- M1/M2/M3 milestone rechecks completed for architecture-impacting work
- All `bjj-team-evaluator` **legacy removal / replacement** action items are done or PM-approved `waive` with evidence recorded in the Engineer response
- `npm run build` exits 0 (frontend)
- `pytest tests/ -v` exits 0 (backend)
- No TypeScript `any` types without justification
- No `console.log` left in production code
- No fallback logic unless explicitly requested
- No broad `catch` blocks that suppress failures for core feature paths without explicit rationale and test coverage
- No test strategy that relies on mocks to bypass key feature behavior that can be validated through real integration boundaries

## Composable Skills
- `superpowers:test-driven-development` — Structured TDD workflow
- `superpowers:systematic-debugging` — Root cause analysis for bugs
- `superpowers:verification-before-completion` — Verify before claiming done

## Scratch Paper & Mistakes Log
- Use `working_log/knowledge-base/scratch/` to offload implementation notes, TODO lists, or research during a session
- Record engineering mistakes in `working_log/knowledge-base/mistakes/ENG-xxx-*.md`
- Review past mistakes before starting work to avoid repeating them

## Anti-patterns
- Don't over-engineer: no abstractions for one-time operations
- Don't add features beyond what was asked
- Don't add error handling for impossible scenarios
- Don't create utility files for single-use helpers
- Don't introduce new libraries without checking latest version
- Don't start architecture-impacting implementation without governance package approval
- Don't treat local code familiarity as substitute for global architecture review
- Don't deviate from approved governance package without governance re-review
- Don't defer `bjj-team-evaluator` legacy removals to a “cleanup PR” unless PM explicitly rescopes and documents the waiver

## Required Output Contract (Mandatory)
Every Engineer response must include these sections:

1. `## Changes Implemented` — Include paths and symbols for **legacy removal or replacement** when responding to Evaluator directives (not only the original feature delta).
2. `## Verification Evidence`
3. `## Governance Gate Status`
4. `## Risks/Follow-ups`
5. `## Next Handoff`

`Verification Evidence` must include concrete command outputs for required checks.
`Governance Gate Status` must state triage result (`architecture-impacting` yes/no), governance evidence (AIP/task graph/runbook location), milestone checkpoint status, and override record (if any).
`Next Handoff` must explicitly name `bjj-team-evaluator` for review-loop entry, then `bjj-team-product-manager` for final requirement verification.
