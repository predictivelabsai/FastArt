"""FastArt interactive self-attention explainer (FastHTML + HTMX).

Run from the repo root (after `uv sync`):
    uv run python web/main.py

Then open http://127.0.0.1:5001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_STATIC = Path(__file__).resolve().parent / "static"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fasthtml.common import (
    A,
    Button,
    Code,
    Div,
    Form,
    H1,
    H2,
    Link,
    Main,
    NotStr,
    P,
    Script,
    Textarea,
    Title,
    fast_app,
    serve,
)

from fastart.attention import compute_self_attention
from fastart.export import attention_to_svg

DEFAULT_SENTENCE = "the cat sat on the mat"

_CSS = (_STATIC / "demo.css").read_text(encoding="utf-8")
_JS = (_STATIC / "demo.js").read_text(encoding="utf-8")

app, rt = fast_app(
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css",
        ),
        # Inline so `python web/main.py` works without a static mount dance.
        NotStr(f"<style>\n{_CSS}\n</style>"),
        NotStr(f"<script>\n{_JS}\n</script>"),
    )
)


def explainer_panel(sentence: str, highlight: int | None = 0):
    result = compute_self_attention(sentence)
    hl = 0 if highlight is None else max(0, min(highlight, len(result.tokens) - 1))
    svg = attention_to_svg(result, highlight=hl)
    weights_json = json.dumps(result.weights_list())

    tokens = Div(
        *[
            Button(
                tok,
                cls=f"tok{' active' if hl == i else ''}",
                type="button",
                onclick=f"selectToken({i})",
            )
            for i, tok in enumerate(result.tokens)
        ],
        cls="tokens",
    )

    arcs = NotStr(
        '<svg id="attn-arcs" class="arc-svg" role="img" '
        'aria-label="Attention arcs from focused query"></svg>'
    )

    return Div(
        P(
            "Click a token to focus its query row. Arc stroke ∝ weight; "
            "hover the grid for exact q→k values.",
            id="focus-hint",
            cls="figure-caption",
        ),
        tokens,
        Div(
            Div(
                P("Links", cls="panel-label"),
                Div(arcs, cls="arcs-wrap"),
            ),
            Div(
                P("Heatmap", cls="panel-label"),
                Div(NotStr(svg), cls="heatmap-wrap", id="heatmap"),
            ),
            cls="viz-row",
        ),
        NotStr(f'<script type="application/json" id="attn-data">{weights_json}</script>'),
        P(
            f"{len(result.tokens)} tokens · d_model={result.d_model} · d_k={result.d_k} · "
            "toy untrained Q/K/V (numpy).",
            cls="meta",
        ),
        id="explainer",
        cls="figure",
    )


@rt("/")
def get():
    return (
        Title("FastArt — Self-Attention Explainer"),
        Main(
            Div(
                Div(
                    P("FastArt · v1", cls="kicker"),
                    H1("Seeing self-attention", cls="brand"),
                    P(
                        "An open Distill-style explainer for the mechanism behind "
                        "transformers. Edit a sentence, focus a query token, and watch "
                        "how attention mass moves across keys — same core as the CLI.",
                        cls="lede",
                    ),
                    cls="essay",
                ),
                H2("The recipe"),
                Div(
                    P(
                        "Every token may look at every other token. For each "
                        "query we score keys with a scaled dot product, turn those "
                        "scores into a distribution with softmax, and mix values "
                        "with those weights:"
                    ),
                    P("Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V", cls="formula"),
                    P(
                        "Q, K, and V are linear projections of the same embeddings "
                        "(self-attention). √dₖ keeps dot products from exploding with "
                        "dimension so softmax does not collapse to a spike. "
                        "Projections here are deterministic toys — for teaching the "
                        "geometry, not modeling language."
                    ),
                    cls="prose essay",
                ),
                Form(
                    Div(
                        Textarea(
                            DEFAULT_SENTENCE,
                            name="sentence",
                            rows=2,
                            maxlength=200,
                            aria_label="Sentence",
                        ),
                        Button("Update figure", type="submit"),
                        cls="row",
                    ),
                    hx_post="/explain",
                    hx_target="#explainer",
                    hx_swap="outerHTML",
                    cls="sentence-form",
                ),
                explainer_panel(DEFAULT_SENTENCE, highlight=0),
                Div(
                    P(
                        A("GitHub", href="https://github.com/predictivelabsai/FastArt"),
                        " · ",
                        A("Inspiration", href="/inspiration"),
                        " · CLI: ",
                        Code('fastart attention "your sentence"'),
                    ),
                    cls="site",
                ),
                cls="site-wrap",
            ),
        ),
    )


@rt("/explain")
def post(sentence: str = DEFAULT_SENTENCE):
    text = (sentence or DEFAULT_SENTENCE).strip() or DEFAULT_SENTENCE
    return explainer_panel(text, highlight=0)


@rt("/inspiration")
def inspiration():
    return (
        Title("Inspiration — FastArt"),
        Main(
            Div(
                P(A("← Back", href="/")),
                H1("Inspiration", cls="brand"),
                P("Full list: INSPIRATION.md in the repository.", cls="prose"),
                Div(
                    P(
                        A(
                            "Transformer Explainer (Polo Club)",
                            href="https://poloclub.github.io/transformer-explainer/",
                            target="_blank",
                        )
                    ),
                    P(
                        A(
                            "Bycroft LLM visualization",
                            href="https://bbycroft.net/llm",
                            target="_blank",
                        )
                    ),
                    P(
                        A(
                            "The Illustrated Transformer",
                            href="https://jalammar.github.io/illustrated-transformer/",
                            target="_blank",
                        )
                    ),
                    P(
                        A(
                            "Distill — Attention and Augmented RNNs",
                            href="https://distill.pub/2016/augmented-rnns/",
                            target="_blank",
                        )
                    ),
                    cls="prose",
                ),
                cls="site-wrap essay",
            ),
        ),
    )


if __name__ == "__main__":
    serve(port=5001)
