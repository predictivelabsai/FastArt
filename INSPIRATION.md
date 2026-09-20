# Inspiration

FastArt aims for the clarity of Distill-style explainers with a small, hackable
Python stack (FastHTML + numpy). v1 code is **original** — we do not copy
upstream visualizer source.

## Key links

| Resource | Why it matters |
|----------|----------------|
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club / Georgia Tech) | Interactive GPT-2-style walkthrough of attention and the full stack |
| [LLM Visualization](https://bbycroft.net/llm) (Brendan Bycroft) | 3D / mechanistic view of transformer internals |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar) | Classic prose + diagrams for Q/K/V and multi-head attention |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original transformer paper — `softmax(QKᵀ/√dₖ)V` |
| [Distill — Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/) | Early Distill treatment of attention |
| [Distill — Visualizing Neural Networks](https://distill.pub/) | Editorial bar for interactive ML explainers |
| [BertViz](https://github.com/jessevig/bertviz) | Attention-head visualization patterns |
| [FastHTML](https://fastht.ml/) | Web framework used for the demo site |

## Design notes for contributors

- Prefer **short Distill-like prose** next to the interactive widget.
- Keep the **same core** for web and CLI (`fastart.attention`).
- Stay educational: label toy / untrained maths clearly.
- SVG-first heatmaps; light vanilla JS is fine; D3 via CDN is optional later.
