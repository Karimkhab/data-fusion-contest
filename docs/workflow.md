# Team Workflow

## Branching
- `main` is protected/stable.
- Each person works in own branch: `feat/<owner>/<topic>`.
- No direct pushes to `main`; only pull requests.

## Notebooks policy
- One owner per notebook path:
  - `notebooks/karim/*`
  - `notebooks/friend/*`
- Do not co-edit a single `.ipynb` file.

## Commit style
- Use prefixes: `feat:`, `fix:`, `exp:`, `docs:`, `chore:`.

## PR checklist
- Rebase on latest `main`.
- Keep PR focused on one logical change.
- Update experiment registry if model/feature behavior changed.
