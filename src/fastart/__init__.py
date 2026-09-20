"""FastArt — Distill-style interactive ML explainers."""

from .attention import AttentionResult, compute_self_attention, tokenize

__version__ = "0.1.0"
__all__ = ["AttentionResult", "compute_self_attention", "tokenize", "__version__"]
