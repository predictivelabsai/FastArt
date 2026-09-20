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

Distill-ish essay layout with an interactive figure: click a token to focus its
query row (non-focus rows dim), hover cells for q→k tooltips, and inspect
attention arcs beside the heatmap. Submit the form to recompute (HTMX partial
swap). CLI SVG export shares the richer heatmap (colorbar + axis labels).

## Layout

```
README.md  ROADMAP.md  INSPIRATION.md  LICENSE  pyproject.toml
src/fastart/
  __init__.py
  attention.py   # toy Q/K/V + softmax attention
  export.py      # SVG heatmap (+ colorbar) + JSON
  cli.py         # `fastart attention`
web/
  main.py        # FastHTML + HTMX demo
  static/        # Distill-ish CSS + focus/arcs/tooltip JS
```

## License

MIT — see [LICENSE](LICENSE).
