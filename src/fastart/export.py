"""Export attention results to SVG heatmaps and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from .attention import AttentionResult


def attention_to_json(result: AttentionResult, *, indent: int = 2) -> str:
    return json.dumps(result.to_dict(), indent=indent)


def write_json(result: AttentionResult, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(attention_to_json(result) + "\n", encoding="utf-8")
    return out


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _cell_color(w: float) -> str:
    """Map weight in [0, 1] to a clean white → indigo sequential scale.

    Stops approximate a readable heat ramp (low = near-white, high = deep indigo).
    """
    w = max(0.0, min(1.0, float(w)))
    # Multi-stop: #ffffff → #c7d2fe → #6366f1 → #312e81
    stops = [
        (0.0, (255, 255, 255)),
        (0.25, (199, 210, 254)),
        (0.55, (99, 102, 241)),
        (1.0, (49, 46, 129)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if w <= t1 or i == len(stops) - 2:
            u = 0.0 if t1 == t0 else (w - t0) / (t1 - t0)
            u = max(0.0, min(1.0, u))
            r = int(_lerp(c0[0], c1[0], u))
            g = int(_lerp(c0[1], c1[1], u))
            b = int(_lerp(c0[2], c1[2], u))
            return f"rgb({r},{g},{b})"
    return "rgb(49,46,129)"


def _short_label(tok: str, limit: int = 10) -> str:
    t = tok if len(tok) <= limit else tok[: limit - 1] + "…"
    return escape(t)


def attention_to_svg(
    result: AttentionResult,
    *,
    cell: int = 36,
    label_w: int = 96,
    highlight: int | None = None,
    colorbar: bool = True,
) -> str:
    """Build an SVG attention heatmap (query rows × key columns).

    Includes a sequential colorbar, axis captions, and data-* attributes so a
    web page can drive focus / tooltips without regenerating the figure.
    """
    tokens = result.tokens
    n = len(tokens)
    pad = 12
    title_h = 30
    axis_gap = 18
    cbar_w = 56 if colorbar else 0
    top_label_h = 52
    bottom_pad = 28

    grid_w = n * cell
    grid_h = n * cell
    width = pad + label_w + grid_w + cbar_w + pad
    height = title_h + axis_gap + top_label_h + grid_h + bottom_pad + pad

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Self-attention heatmap" '
        f'class="attn-svg" data-n="{n}">',
        "<style>",
        ".lbl{font:11px 'Source Sans 3',ui-sans-serif,system-ui,sans-serif;fill:#334155}",
        ".axis{font:600 11px 'Source Sans 3',ui-sans-serif,system-ui,sans-serif;fill:#64748b}",
        ".title{font:600 13px 'Source Sans 3',ui-sans-serif,system-ui,sans-serif;fill:#0f172a}",
        ".cbar-lbl{font:10px 'Source Sans 3',ui-sans-serif,system-ui,sans-serif;fill:#64748b}",
        ".cell{stroke:#cbd5e1;stroke-width:0.6;transition:opacity .22s ease,stroke .22s ease,stroke-width .22s ease}",
        ".cell.hi{stroke:#f59e0b;stroke-width:2.4}",
        ".cell.dim{opacity:0.28}",
        ".row-lbl{transition:opacity .22s ease,fill .22s ease}",
        ".row-lbl.dim{opacity:0.35}",
        ".row-lbl.focus{fill:#312e81;font-weight:700}",
        ".wt{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#1e293b;"
        "pointer-events:none;transition:opacity .22s ease}",
        ".wt.dim{opacity:0.2}",
        "</style>",
        f'<rect width="100%" height="100%" fill="#fafbfc"/>',
        f'<text class="title" x="{pad}" y="20">Attention weights · softmax(QKᵀ/√dₖ)</text>',
    ]

    origin_x = pad + label_w
    origin_y = title_h + axis_gap + top_label_h

    # Axis captions
    parts.append(
        f'<text class="axis" x="{origin_x + grid_w / 2}" y="{title_h + 14}" '
        f'text-anchor="middle">Key →</text>'
    )
    parts.append(
        f'<text class="axis" x="{pad + 10}" y="{origin_y + grid_h / 2}" '
        f'text-anchor="middle" transform="rotate(-90 {pad + 10} {origin_y + grid_h / 2})">'
        f"Query ↓</text>"
    )

    # Column labels (keys) — rotated, readable
    for j, tok in enumerate(tokens):
        x = origin_x + j * cell + cell / 2
        y = origin_y - 10
        parts.append(
            f'<text class="lbl col-lbl" data-k="{j}" x="{x}" y="{y}" text-anchor="end" '
            f'transform="rotate(-40 {x} {y})">{_short_label(tok)}</text>'
        )

    # Cells + row labels
    for i, tok in enumerate(tokens):
        y = origin_y + i * cell
        focus_cls = " focus" if highlight is not None and i == highlight else ""
        parts.append(
            f'<text class="lbl row-lbl{focus_cls}" data-q="{i}" '
            f'x="{origin_x - 8}" y="{y + cell / 2 + 4}" text-anchor="end">'
            f"{_short_label(tok, 12)}</text>"
        )
        for j in range(n):
            w = float(result.weights[i, j])
            x = origin_x + j * cell
            hi = " hi" if highlight is not None and i == highlight else ""
            parts.append(
                f'<rect class="cell{hi}" data-q="{i}" data-k="{j}" data-w="{w:.6f}" '
                f'x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_cell_color(w)}">'
                f"<title>{escape(tok)} → {escape(tokens[j])}: {w:.3f}</title></rect>"
            )
            if w >= 0.08 and cell >= 28:
                parts.append(
                    f'<text class="wt" data-q="{i}" data-k="{j}" '
                    f'x="{x + cell / 2}" y="{y + cell / 2 + 3}" text-anchor="middle">'
                    f"{w:.2f}</text>"
                )

    if colorbar:
        cb_x = origin_x + grid_w + 18
        cb_y = origin_y
        cb_h = grid_h
        cb_bar_w = 14
        steps = 48
        for s in range(steps):
            t = 1.0 - (s + 0.5) / steps
            sy = cb_y + s * (cb_h / steps)
            sh = cb_h / steps + 0.5
            parts.append(
                f'<rect x="{cb_x}" y="{sy}" width="{cb_bar_w}" height="{sh}" '
                f'fill="{_cell_color(t)}" stroke="none"/>'
            )
        parts.append(
            f'<rect x="{cb_x}" y="{cb_y}" width="{cb_bar_w}" height="{cb_h}" '
            f'fill="none" stroke="#94a3b8" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text class="cbar-lbl" x="{cb_x + cb_bar_w + 4}" y="{cb_y + 9}">high</text>'
        )
        parts.append(
            f'<text class="cbar-lbl" x="{cb_x + cb_bar_w + 4}" y="{cb_y + cb_h}">low</text>'
        )
        parts.append(
            f'<text class="cbar-lbl" x="{cb_x + cb_bar_w / 2}" y="{cb_y + cb_h + 16}" '
            f'text-anchor="middle">attention</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(
    result: AttentionResult,
    path: str | Path,
    *,
    highlight: int | None = None,
) -> Path:
    out = Path(path)
    out.write_text(attention_to_svg(result, highlight=highlight) + "\n", encoding="utf-8")
    return out
