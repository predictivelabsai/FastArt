# FastArt

Open toolkit for **Distill-style interactive ML explainers**.

- **FastHTML** interactive demo site (HTMX + SVG heatmap)
- **CLI** sharing the same numpy toy self-attention core
- **Metal prints** — documented on the roadmap only (no ordering in v1)

## v1: Self-attention explainer

Edit a short sentence, inspect token→token attention weights from
`softmax(Q Kᵀ / √d) V`, and export SVG/JSON from the CLI.

Projections are **toy and untrained** (deterministic from a seed) — for teaching
the mechanism, not for modeling language.

Inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
and [Bycroft’s LLM viz](https://bbycroft.net/llm); implementation is original
(see [INSPIRATION.md](INSPIRATION.md)).

## Install

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) (recommended).

```bash
git clone https://github.com/predictivelabsai/FastArt.git
cd FastArt
uv sync
```

Or with pip:

```bash
pip install -e .
```

## CLI

```bash
# Help
uv run fastart attention --help

# Export SVG + JSON (default sentence)
uv run fastart attention

# Custom sentence → files
uv run fastart attention "attention is all you need" -o out/attn --format both

# SVG only, highlight query row 2
uv run fastart attention "the cat sat" --format svg --highlight 2
```

## Interactive site

```bash
uv run python web/main.py
# → http://127.0.0.1:5001
```

Click a token to highlight its attention row. Submit the form to recompute for a
new sentence (HTMX partial swap).

## Layout

```
README.md  ROADMAP.md  INSPIRATION.md  LICENSE  pyproject.toml
src/fastart/
  __init__.py
  attention.py   # toy Q/K/V + softmax attention
  export.py      # SVG heatmap + JSON
  cli.py         # `fastart attention`
web/main.py      # FastHTML + HTMX demo
```

## License

MIT — see [LICENSE](LICENSE).
