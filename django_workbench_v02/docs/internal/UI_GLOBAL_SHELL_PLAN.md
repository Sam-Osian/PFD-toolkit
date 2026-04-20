# UI Global Shell Plan (Step 1)

## Goal
Build a single, reusable site shell for v0.2 (layout, navigation, design tokens, typography, spacing, reusable primitives) before page-by-page UI porting.

This stage is visual-only. It must not change business logic, permissions, run orchestration, or URLs.

## Status
- Stage 1 (global shell): complete.
- Stage 2 (page-by-page port): complete (Dashboard, Investigation Wizard, Investigation Detail + Run Detail, Collections, Auth, and Shared/Public pages ported).
- Stage 3 (static asset plumbing): complete for web runtime.
- Stage 4 (shared includes for nav/messages): complete.

## Constraints
- Keep backend contracts locked per `docs/internal/BACKEND_CONTRACTS_UI_PORT.md`.
- Preserve all existing routes and form actions.
- No feature additions in this step.
- Mobile-first responsive behavior is required.

## Source of truth for look-and-feel
- `claude_design/PFD Toolkit/shared/tokens.css`
- `claude_design/PFD Toolkit/prototype/app.css`
- `claude_design/PFD Toolkit/_reference/templates/base.html`
- `claude_design/PFD Toolkit/brand/logo-lens.svg`

## Target files (Step 1)

### 1) Base template shell
- `templates/base.html`

Changes:
- Replace inline `<style>` with static CSS includes.
- Add explicit layout regions:
  - app background wrapper
  - top navigation shell
  - content container
  - optional page header block
- Add template blocks for extensibility:
  - `block page_title`
  - `block page_actions`
  - `block content`
  - optional `block extra_head` and `block extra_js`
- Keep existing nav link behavior and permission gates.
- Keep existing Django messages rendering, but style via component classes.

### 2) Global stylesheet and tokens
- `static/css/tokens.css` (new)
- `static/css/base.css` (new)
- `static/css/components.css` (new)

Changes:
- Extract semantic tokens (color, spacing, radius, shadows, durations).
- Define typography scale and font stack for headings/body/code.
- Define common primitives:
  - `.container`, `.stack`, `.cluster`
  - `.card`, `.panel`, `.surface`
  - `.btn` variants (`primary`, `secondary`, `ghost`, `danger`)
  - `.badge`, `.pill`, `.table-wrap`
  - message styles (`success`, `warning`, `error`, `info`)
- Add responsive breakpoints and reduced-motion handling.

### 3) Static asset plumbing
- `pfd_workbench_v02/settings.py` (only if needed)

Changes:
- Keep `STATIC_URL` as-is.
- Add `STATICFILES_DIRS = [BASE_DIR / "static"]` if not already present and needed for project-level static assets.

### 4) Shared includes (optional but recommended)
- `templates/includes/nav.html` (new)
- `templates/includes/messages.html` (new)

Changes:
- Move nav and messages partials out of `base.html` for maintainability.
- Keep behavior identical; styling only.

## Non-goals in Step 1
- No wizard field/layout rewrites yet.
- No dashboard card logic changes.
- No collection query/data loading changes.
- No run detail behavior changes.

## Implementation sequence
1. Add static token/base/component CSS files.
2. Refactor `base.html` to consume these styles.
3. Apply minimal utility classes to keep current pages readable under new shell.
4. Smoke-check core pages:
   - `/`
   - `/dashboard/`
   - `/collections/`
   - one workbook detail page
   - one investigation wizard page
   - one run detail page
5. Fix regressions in spacing/wrapping without changing page logic.

## Acceptance checks (Step 1 done when all pass)
- Every template extending `base.html` renders without HTML errors.
- Navigation behavior is unchanged for authenticated vs anonymous users.
- Flash/messages are visible and styled.
- No form action/CSRF regressions.
- Mobile (<= 768px) is usable for nav + primary content.
- Lighthouse/UX sanity: no blocking layout shift from shell CSS load.

## Risks and mitigations
- Risk: style collisions with existing inline/table markup.
  - Mitigation: scope selectors under shell root classes and prefer utility classes.
- Risk: unintentionally changing interaction affordances.
  - Mitigation: preserve button/link semantics and run page-by-page smoke checks.
- Risk: backend drift during UI pass.
  - Mitigation: do not edit view/service logic in this stage.

## Next step after Step 1
Page-by-page port in this order:
1. Dashboard (`templates/wb_workspaces/dashboard.html`)
2. Investigation wizard (`templates/wb_investigations/investigation_wizard.html`)
3. Investigation detail + run timeline (`templates/wb_investigations/investigation_detail.html`, `templates/wb_runs/run_detail.html`)
4. Collections list/detail (`templates/wb_collections/collection_list.html`, `templates/wb_collections/collection_detail.html`)
5. Auth and shared/public pages
