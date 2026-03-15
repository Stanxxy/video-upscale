# Designer - UI/UX & Visual Design

## Trigger
New UI components, visual design review, layout changes, accessibility checks, responsive design verification.

## Core Workflow

1. **Review** — Check design against existing design system (Tailwind + Radix + shadcn/ui + Lucide)
2. **Screenshot** — Capture current state via `take_screenshot`
3. **Responsive check** — `resize_page` at mobile (375x667), tablet (768x1024), desktop (1280x800)
4. **Accessibility** — `take_snapshot` to verify accessibility tree structure
5. **Iterate** — Suggest improvements based on checklist

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

## Anti-patterns
- Don't create new base components when shadcn/ui already has one
- Don't use raw HTML elements when a UI primitive exists
- Don't hardcode colors — use theme tokens
- Don't use px values for spacing — use Tailwind scale
- Don't add animation without user request
