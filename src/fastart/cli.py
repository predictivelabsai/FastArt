"""FastArt command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .attention import compute_self_attention
from .export import write_json, write_svg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastart",
        description="FastArt — Distill-style ML explainers (CLI + FastHTML site).",
    )
    p.add_argument("--version", action="version", version=f"fastart {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    att = sub.add_parser(
        "attention",
        help="Compute toy self-attention and export SVG and/or JSON.",
    )
    att.add_argument(
        "text",
        nargs="?",
        default="the cat sat on the mat",
        help="Sentence to explain (default: demo sentence).",
    )
    att.add_argument(
        "-o",
        "--output",
        default="attention",
        help="Output path prefix without extension (default: attention).",
    )
    att.add_argument(
        "--format",
        choices=("svg", "json", "both"),
        default="both",
        help="Export format (default: both).",
    )
    att.add_argument("--d-model", type=int, default=32, help="Embedding size.")
    att.add_argument("--d-k", type=int, default=16, help="Key/query/value size.")
    att.add_argument("--seed", type=int, default=42, help="Projection seed.")
    att.add_argument(
        "--highlight",
        type=int,
        default=None,
        metavar="I",
        help="Highlight query row index in the SVG.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "attention":
        result = compute_self_attention(
            args.text,
            d_model=args.d_model,
            d_k=args.d_k,
            seed=args.seed,
        )
        prefix = Path(args.output)
        written: list[str] = []
        if args.format in ("svg", "both"):
            path = write_svg(result, prefix.with_suffix(".svg"), highlight=args.highlight)
            written.append(str(path))
        if args.format in ("json", "both"):
            path = write_json(result, prefix.with_suffix(".json"))
            written.append(str(path))

        print(f"tokens: {' | '.join(result.tokens)}")
        print(f"shape:  weights {result.weights.shape}  (d_model={result.d_model}, d_k={result.d_k})")
        for w in written:
            print(f"wrote:  {w}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
