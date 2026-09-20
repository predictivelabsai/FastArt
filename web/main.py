"""FastArt interactive self-attention explainer (FastHTML + HTMX).

Run from the repo root (after `uv sync`):
    uv run python web/main.py

Then open http://127.0.0.1:5001
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
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
    Span,
    Style,
    Textarea,
    Title,
    fast_app,
    serve,
)

from fastart.attention import compute_self_attention
from fastart.export import attention_to_svg

DEFAULT_SENTENCE = "the cat sat on the mat"

app, rt = fast_app(
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css",
        ),
        Style(
            """
            :root { --fa-accent: #4f46e5; }
            body { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
            .prose code { background: #eef2ff; padding: 0.1em 0.35em; border-radius: 4px; }
            .tokens { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.75rem 0; }
            .tok {
              border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.35rem 0.75rem;
              cursor: pointer; background: #fff; font-weight: 500;
            }
            .tok:hover, .tok.active { border-color: var(--fa-accent); background: #eef2ff; }
            .heatmap-wrap {
              overflow-x: auto; background: #f8fafc; border-radius: 12px;
              padding: 0.75rem; border: 1px solid #e2e8f0;
            }
            .formula {
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              background: #0f172a; color: #e2e8f0; padding: 0.85rem 1rem;
              border-radius: 10px; overflow-x: auto;
            }
            footer.site { margin-top: 2.5rem; color: #64748b; font-size: 0.9rem; }
            """
        ),
        Script(
            """
            function selectToken(i) {
              document.querySelectorAll('.tok').forEach((el, idx) => {
                el.classList.toggle('active', idx === i);
              });
              const svg = document.querySelector('#heatmap svg');
              if (!svg) return;
              svg.querySelectorAll('rect.cell').forEach(r => r.classList.remove('hi'));
              const n = document.querySelectorAll('.tok').length;
              const rects = svg.querySelectorAll('rect.cell');
              for (let j = 0; j < n; j++) {
                const idx = i * n + j;
                if (rects[idx]) rects[idx].classList.add('hi');
              }
              const hint = document.getElementById('focus-hint');
              const t = document.querySelectorAll('.tok')[i];
              if (hint && t) {
                hint.textContent = 'Focusing query token “' + t.textContent.trim()
                  + '” — amber border marks its attention row.';
              }
            }
            """
        ),
    )
)


def explainer_panel(sentence: str, highlight: int | None = 0):
    result = compute_self_attention(sentence)
    hl = 0 if highlight is None else highlight
    svg = attention_to_svg(result, highlight=hl)
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
    return Div(
        P(
            "Click a token to highlight its attention row (what it looks at).",
            id="focus-hint",
        ),
        tokens,
        Div(NotStr(svg), cls="heatmap-wrap", id="heatmap"),
        P(
            f"{len(result.tokens)} tokens · d_model={result.d_model} · d_k={result.d_k} · "
            "toy untrained Q/K/V (numpy).",
            style="color:#64748b;font-size:0.85rem;",
        ),
        id="explainer",
    )


@rt("/")
def get():
    return (
        Title("FastArt — Self-Attention Explainer"),
        Main(
            H1("FastArt"),
            P(
                "Open toolkit for Distill-style interactive ML explainers. "
                "v1: a hands-on self-attention sandbox — edit a sentence, watch "
                "token→token attention, export the same core from the CLI."
            ),
            H2("Self-attention"),
            Div(
                P(
                    "Transformers let every token look at every other token. "
                    "For each query token we score keys with a scaled dot product, "
                    "turn those scores into a probability distribution with softmax, "
                    "and mix values with those weights:"
                ),
                P("Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V", cls="formula"),
                P(
                    "Here Q, K, and V are linear projections of the same token embeddings "
                    "(self-attention). √dₖ keeps the dot products from growing with "
                    "dimension so softmax does not collapse to a one-hot spike. "
                    "This demo uses deterministic toy projections — educational, not a trained model.",
                    cls="prose",
                ),
                cls="prose",
            ),
            Form(
                Textarea(
                    DEFAULT_SENTENCE,
                    name="sentence",
                    rows=2,
                    maxlength=200,
                    aria_label="Sentence",
                ),
                Button("Update", type="submit"),
                hx_post="/explain",
                hx_target="#explainer",
                hx_swap="outerHTML",
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
            H1("Inspiration"),
            P(A("← Back", href="/")),
            P("Full list: INSPIRATION.md in the repository."),
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
        ),
    )


if __name__ == "__main__":
    serve(port=5001)
