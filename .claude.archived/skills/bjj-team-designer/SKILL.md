---
name: bjj-team-designer
description: Provides UI/UX and visual design review guidance for BJJ Vision interfaces. Use when creating or refining frontend components, layouts, responsiveness, and accessibility.
when_to_use: Trigger for frontend UI/UX changes, layout adjustments, responsiveness checks, and accessibility review before implementation sign-off.
paths:
  - bjj-vision-frontend/**
  - bjj-video-analyzer/**
  - working_log/**
user-invocable: true
---

# Designer - UI/UX & Visual Design

## Trigger
New UI components, visual design review, layout changes, accessibility checks, responsive design verification.

## Knowledge Base Preflight (Always-On)
Before any design workflow step:

1. Read `working_log/knowledge-base/INDEX.md`.
2. Query relevant entries for design/UX/accessibility context.
3. Extract prior decisions, known pitfalls, and unresolved design issues.
4. If no relevant entry exists, proceed and note the KB gap for possible post-task capture.

## KB Update Triggers (Moderate Auto-Create)
Create or update a KB candidate when any of these are true:
- A plan is accepted and changes user-visible UI behavior, navigation, or interaction patterns.
- A nontrivial design bug is fixed (layout break, accessibility regression, responsive failure) with reproducible evidence.

Skip KB updates for trivial style tweaks (spacing typo-level edits, wording-only polish, no behavior impact).

When a KB update qualifies:
- Capture evidence (affected files, screenshots, snapshots, acceptance notes).
- Hand off lifecycle ownership to `bjj-team-meta` for alignment checks and safe-auto maintenance.

## Core Workflow

1. **Review** — Check design against existing design system (Tailwind + Radix + shadcn/ui + Lucide)
2. **Screenshot** — Capture current state via `take_screenshot`
3. **Responsive check** — `resize_page` at mobile (375x667), tablet (768x1024), desktop (1280x800)
4. **Accessibility** — `take_snapshot` to verify accessibility tree structure
5. **Iterate** — Suggest improvements based on checklist
6. **Evaluator loop** — Enter the mandatory Designer/Engineer <-> Evaluator loop until all findings are closed with evidence and explicit agreement.

## Design System

### Component Library
- **Primitives**: `src/components/ui/` (shadcn/ui + Radix UI)
- **Always reuse** existing primitives before creating new ones
- **Check** `src/components/ui/` for: Button, Input, Dialog, Select, Tabs, Card, Badge, etc.

### Styling
- **Tailwind CSS** utility classes exclusively
- **No custom CSS** unless truly impossible with Tailwind
- **Spacing**: Use Tailwind spacing scale (p-2, p-4, gap-3, etc.)
- **Colors**: Use theme tokens (primary, secondary, muted, destructive, etc.)

### Icons
- **Lucide React** exclusively — no mixing icon libraries
- Import from `lucide-react`

### Typography
- Use Tailwind text utilities (text-sm, text-lg, font-medium, etc.)
- Headings: `text-2xl font-bold`, `text-xl font-semibold`, etc.

## Checklist
- [ ] Uses existing `src/components/ui/` primitives (no reinventing)
- [ ] Responsive at mobile / tablet / desktop breakpoints
- [ ] Color contrast meets WCAG AA (4.5:1 text, 3:1 large text)
- [ ] Interactive elements have visible focus states
- [ ] Icons from Lucide React only
- [ ] Consistent spacing using Tailwind scale
- [ ] Loading and empty states designed
- [ ] Error states have clear messaging
- [ ] Dark mode compatible (if applicable)
- [ ] No horizontal overflow at any breakpoint

## Viewport Breakpoints
```
sm: 640px    — small phones landscape
md: 768px    — tablets
lg: 1024px   — small laptops
xl: 1280px   — desktops
2xl: 1536px  — large screens
```

## Self-Critique Gate
Before signing off, ask: **"Would this pass as a well-designed, popular app?"**
- Compare against apps like Linear, Notion, Vercel Dashboard
- Does the content fill the viewport? Are there awkward gaps?
- Is spacing consistent? Do elements align properly?
- If the answer is no, don't sign off — file it and fix it

## Scratch Paper & Mistakes Log
- Use `working_log/knowledge-base/scratch/` for intermediate design notes during a session
- Record design misses in `working_log/knowledge-base/mistakes/DESIGN-xxx-*.md`
- Review past mistakes before starting a new design review

## Gates
- WCAG AA compliance on color contrast
- Uses existing component primitives from `src/components/ui/`
- No layout breaks at any standard breakpoint
- **No horizontal overflow at any width (320px+)**
- Accessibility tree (from `take_snapshot`) shows proper roles and labels
- Self-critique gate passes
- Evaluator findings are resolved or explicitly waived with approved rationale

## Anti-patterns
- Don't create new base components when shadcn/ui already has one
- Don't use raw HTML elements when a UI primitive exists
- Don't hardcode colors — use theme tokens
- Don't use px values for spacing — use Tailwind scale
- Don't add animation without user request

## Required Output Contract (Mandatory)
Every Designer response must include these sections:

1. `## UI Findings`
2. `## Accessibility/Responsive Checks`
3. `## Required Fixes`
4. `## Evaluator Loop Status`
5. `## Next Handoff`

`Next Handoff` must explicitly name `bjj-team-engineer` (implementation) or `bjj-team-evaluator` (comprehensive review loop).
