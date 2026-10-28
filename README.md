# ntfz

Physics-informed neural travel-time fields (the NTFields lineage) with a shared problem definition, trainer,
planners and benchmark. See [docs/DESIGN.md](docs/DESIGN.md).

```bash
uv sync                          # dev environment
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```
