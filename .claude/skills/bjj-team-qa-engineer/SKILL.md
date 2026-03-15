# QA Engineer - Testing & Verification

## Trigger
After any implementation, when verifying UI behavior, writing E2E tests, reproducing bugs.

## Screenshot Storage
- All QA screenshots are temporary files — store them in `temp_data/qa-screenshots/`
- Always pass `filename` as `temp_data/qa-screenshots/{name}.png` when calling `take_screenshot`
- Clean up the directory at the end of a QA session or when it accumulates too many files
- This directory is gitignored — never commit screenshots

## Scratch Paper
- Use `working_log/knowledge-base/scratch/` to offload intermediate notes during a session (e.g., list of pages to check, test plan, findings so far)
- This keeps the main context focused on the current task
- Clean up scratch files at session end

## Core Workflow (Frontend - Chrome DevTools MCP)

1. **Navigate** — `navigate_page` to the page under test
2. **Snapshot** — `take_snapshot` to get accessibility tree (structural verification)
3. **Interact** — `click`, `fill_form`, `press_key` to simulate user actions
4. **Screenshot** — `take_screenshot` with `filename: "temp_data/qa-screenshots/{name}.png"` for visual verification
5. **Assert** — Verify expected elements, text, and state in snapshot/screenshot
6. **Resize** — `resize_page` to test responsive behavior at breakpoints (mobile: 320x568, 375x667, tablet: 768x1024, desktop: 1280x800)
7. **Overflow Check** — Run programmatic overflow detection (see below) — screenshots alone are NOT sufficient

## Programmatic Responsive Check (MANDATORY)
Visual screenshots can miss horizontal overflow. Always run this via `evaluate_script`:
```js
// Check all pages at narrow widths for horizontal overflow
document.documentElement.scrollWidth > document.documentElement.clientWidth
```
Test at widths: 320, 375, 480, 640 on every page. If `scrollWidth > clientWidth`, find the overflowing element and fix it.

## Self-Critique Gate
Before signing off, ask: **"Would this pass as a well-designed, popular app?"**
- Compare against apps like Linear, Notion, Vercel Dashboard
- Check: Does the content fill the viewport? Are there awkward gaps? Is spacing consistent?
- If the answer is no, don't sign off — file it as a bug and fix it

## Core Workflow (Backend - pytest)

1. **Navigate** to service directory
2. **Run** `pytest tests/ -v` using the backend `.venv`
3. **Verify** exit code 0
4. **Check** test output for warnings or skipped tests

## Frontend Test Files
- Location: `bjj-vision-frontend/tests/*.spec.ts`
- Follow pattern established in `login-dashboard.spec.ts`
- Use Playwright test runner conventions

## Verification Checklist
- [ ] Happy path works end-to-end
- [ ] Error states display correctly
- [ ] Loading states appear during async operations
- [ ] Form validation triggers on invalid input
- [ ] Navigation flows are correct
- [ ] Responsive layout holds at 320/375/768/1024/1280 (no horizontal overflow)
- [ ] No horizontal scrollbar at any width (programmatic check)
- [ ] Content fills viewport — no awkward gaps or empty space

## Gates
- All tests pass (exit 0)
- Visual screenshot shows expected UI state
- **Programmatic overflow check passes at all widths**
- No console errors in `list_console_messages`
- Accessibility tree (`take_snapshot`) contains expected elements

## Mistakes Log
- Record all QA misses in `working_log/knowledge-base/mistakes/QA-xxx-*.md`
- Review past mistakes before starting a new QA session to avoid repeating them

## Composable Skills
- `superpowers:verification-before-completion` — Gate before claiming work is done

## Common Chrome DevTools MCP Patterns
See `references/browser-patterns.md` for detailed examples.

## Bug Reproduction Flow
1. Navigate to the reported state
2. Screenshot the current (broken) behavior
3. Snapshot the accessibility tree to understand structure
4. Document reproduction steps
5. After fix: re-run same steps to verify resolution
