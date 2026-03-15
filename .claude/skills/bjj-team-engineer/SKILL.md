# Engineer - Implementation

## Trigger
Implementing features, fixing bugs, refactoring code, writing tests.

## Core Workflow

1. **Branch** — Create a feature branch from `develop` before starting work: `git checkout develop && git pull && git checkout -b feature/<name>`. Never commit directly to `develop` or `main`.
2. **Understand** — Read relevant code before changing it. Never modify code you haven't read.
3. **TDD cycle** — Write failing test, implement minimally, verify pass. Compose with `superpowers:test-driven-development` for structured TDD.
4. **Build/lint check** — Run `npm run build` (frontend) or `pytest` (backend) to verify.
5. **Hand off to QA** — After implementation passes build, QA Engineer verifies behavior via browser (frontend) or integration tests (backend).

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

## Gates
- `npm run build` exits 0 (frontend)
- `pytest tests/ -v` exits 0 (backend)
- No TypeScript `any` types without justification
- No `console.log` left in production code
- No fallback logic unless explicitly requested

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
