# Local review quality gate

Human and agent quality bar beyond (or before) GitHub Actions.
Run the relevant section before claiming a PR is ready.

## When to run

- Before every push that changes docs that define product boundaries
- Before every push that changes package code, loaders, renderers, or tests
- After resolving merges with `main`

## Docs-only changes (current foundation)

No package install required yet.

```bash
# Charter and agent docs resolve
test -f docs/CHARTER.md
test -f AGENTS.md
test -f REVIEW.md

# README points at the charter
grep -q 'docs/CHARTER.md' README.md

# Basic markdown presence (no broken empty stubs)
test -s docs/CHARTER.md && test -s AGENTS.md && test -s REVIEW.md && test -s README.md
```

**Verdict template:**

```markdown
## Local verification (REVIEW.md)

**Branch tip:** `<sha>`
**Verdict:** Docs gate passed.

| Check | Result |
|-------|--------|
| `docs/CHARTER.md` present | pass |
| `AGENTS.md` / `REVIEW.md` present | pass |
| README links charter | pass |
```

## Package + tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -c "import spike_viz; print(spike_viz.__version__)"
pytest -q
```

Golden export fixture must load:

```bash
python -c "from spike_viz import load_axon_export; print(load_axon_export('fixtures/axon-encoder/rate/tiny_synthetic').meta['encoder'])"
```

Optional formatting / lint (only if configured in the repo):

```bash
# e.g. ruff check . && ruff format --check .
```

### How to read results

- CPU tests must pass on machines without NVIDIA GPUs.
- CUDA paths: skip when no device; never require GPU for default `pytest`.
- Failures that invent data or hide load errors are product bugs — fix, don’t weaken tests.
- Missing fixture files must raise `SpikeIOError`, not return empty spikes.

## PR comment style

- Lead with **branch tip SHA** and pass/fail verdict
- Small tables (Check → Result), not raw log dumps
- Put long logs in `<details>` only if needed

## Out of scope for this file

- Linear dual-tracking process
- Remote-only CI configuration (document in workflow files when #13 lands)
