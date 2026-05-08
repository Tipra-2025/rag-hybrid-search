# Contributing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check
```

`make check` runs `ruff` + `mypy --strict` + `pytest --cov`. PRs are blocked until coverage stays at or above **80 %**.

## Conventions

- **Conventional commits.** `feat(retrieval): ...`, `fix(citation): ...`, `docs(adr): ...`.
- **No untyped public API.** `mypy --strict` will reject it.
- **No real network in tests.** Use `httpx.MockTransport` and the `make_llm_client` fixture.
- **Eval set first.** When you add a new chunking strategy or retrieval tweak, add at least one golden Q&A row that exercises it.
- **Don't blend BM25 and cosine scores directly.** RRF first, score-driven rerank after. (See [ADR-002](docs/ADR-002-rrf-fusion.md).)

## Code of conduct

Be kind. Disagreements are about the work, never about people.
