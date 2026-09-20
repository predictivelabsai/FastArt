# Roadmap

## Shipped in v1

- [x] Toy self-attention core (numpy)
- [x] FastHTML + HTMX interactive heatmap (token click → highlight)
- [x] CLI `fastart attention` → SVG / JSON
- [x] Docs: README, INSPIRATION, LICENSE (MIT)

## Metal prints (document only — do **not** implement ordering in-app)

Future product idea: turn explainer frames (attention maps, embedding grids,
architecture posters) into physical **metal prints**. Research vendors; no cart,
checkout, or print API in the open toolkit until deliberately scoped.

### Estonia / Baltics

- [AluMetalPrint](https://alumetalprint.ee/) — aluminium photo prints
- [Saal Digital](https://www.saal-digital.ee/) — photo products / prints
- [Metallfoto](https://metallfoto.ee/) — metal photo printing

### International

- [WhiteWall](https://www.whitewall.com/) — gallery-quality metal & acrylic
- [AP-Print](https://ap-print.com/) — fine-art / metal options

When we revisit this: export print-ready SVG/PDF at fixed DPI, bleed guides, and
a separate `fastart print-spec` doc — still no payment integration unless
productized separately.

## Later explainers & tooling

- [ ] Full transformer block walkthrough (LN → MHA → FFN)
- [ ] Token embedding / positional encoding playground
- [ ] Multi-head attention comparison view
- [ ] 3D attention / residual stream sketches
- [ ] Blender MCP experiments for spatial explainer assets
- [ ] MonsterUI polish on the FastHTML site
- [ ] Embeddings similarity explorer
- [ ] Packaged demo GIFs for docs

## Non-goals (for now)

- Serving or fine-tuning real LLMs in this repo
- In-app metal-print checkout
