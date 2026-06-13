# Contributing to EchoMind

Thanks for your interest. EchoMind is a portfolio / learning project but PRs and issues are welcome.

## Development setup

```bash
git clone https://github.com/USERNAME/echomind.git
cd echomind
uv sync --all-extras
pre-commit install
```

## Conventions

- **Code style** — `ruff` for lint + format. Pre-commit will enforce on commit.
- **Types** — `mypy` for static checks (non-strict mode). Add type hints to all new public functions.
- **Tests** — `pytest`. New features need a test. Aim for 80%+ coverage on touched files.
- **Commits** — [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- **Branches** — `main` is protected. Use `feat/...`, `fix/...`, `docs/...`.

## PR checklist

- [ ] Tests pass locally (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] Type check passes (`make typecheck`)
- [ ] README / docs updated if behavior changed
- [ ] No secrets / API keys committed
- [ ] PR title follows Conventional Commits

## Running a subset of tests

```bash
# unit only (fast)
uv run pytest -m "not slow and not integration"

# integration (needs running services)
uv run pytest -m integration

# everything except GPU
uv run pytest -m "not gpu"
```
