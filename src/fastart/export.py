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


def _cell_color(w: float) -> str:
    """Map weight in [0,1] to a blue→amber heat color."""
    w = max(0.0, min(1.0, float(w)))
    # light gray → deep indigo
    r = int(240 - w * 180)
    g = int(244 - w * 160)
    b = int(255 - w * 40)
    return f"rgb({r},{g},{b})"


def attention_to_svg(
    result: AttentionResult,
    *,
    cell: int = 36,
    label_w: int = 90,
    highlight: int | None = None,
) -> str:
    """Build an SVG attention heatmap (query rows × key columns)."""
    tokens = result.tokens
    n = len(tokens)
    pad = 8
    title_h = 28
    width = label_w + n * cell + pad * 2
    height = title_h + label_w // 2 + n * cell + pad * 2 + 24

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Self-attention heatmap">'
        f"<style>"
        f".lbl{{font:11px ui-sans-serif,system-ui,sans-serif;fill:#1e293b}}"
        f".title{{font:600 13px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}}"
        f".cell{{stroke:#94a3b8;stroke-width:0.5}}"
        f".hi{{stroke:#f59e0b;stroke-width:2.5}}"
        f"</style>",
        f'<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text class="title" x="{pad}" y="18">Attention weights · softmax(QKᵀ/√d)</text>',
    ]

    origin_x = label_w
    origin_y = title_h + 20

    # column labels (keys)
    for j, tok in enumerate(tokens):
        x = origin_x + j * cell + cell / 2
        y = origin_y - 6
        parts.append(
            f'<text class="lbl" x="{x}" y="{y}" text-anchor="middle" '
            f'transform="rotate(-35 {x} {y})">{escape(tok[:12])}</text>'
        )

    for i, tok in enumerate(tokens):
        y = origin_y + i * cell
        parts.append(
            f'<text class="lbl" x="{origin_x - 6}" y="{y + cell / 2 + 4}" text-anchor="end">'
            f"{escape(tok[:14])}</text>"
        )
        for j in range(n):
            w = float(result.weights[i, j])
            x = origin_x + j * cell
            cls = "cell hi" if highlight is not None and i == highlight else "cell"
            parts.append(
                f'<rect class="{cls}" x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_cell_color(w)}">'
                f"<title>q={escape(tok)} → k={escape(tokens[j])}: {w:.3f}</title></rect>"
            )
            if w >= 0.08:
                parts.append(
                    f'<text class="lbl" x="{x + cell / 2}" y="{y + cell / 2 + 4}" '
                    f'text-anchor="middle" font-size="9">{w:.2f}</text>'
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
