"""Toy self-attention for educational explainers.

Computes softmax(Q K^T / sqrt(d_k)) V with deterministic, untrained
projections — enough to illustrate the mechanism, not a real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Sequence

import numpy as np

DEFAULT_D_MODEL = 32
DEFAULT_D_K = 16


def tokenize(text: str) -> list[str]:
    """Whitespace tokenize; keep punctuation attached for simplicity."""
    tokens = [t for t in text.strip().split() if t]
    return tokens or ["(empty)"]


def _token_vec(token: str, dim: int, salt: bytes) -> np.ndarray:
    """Map a token string to a unit-ish embedding via blake2b + bit unpack."""
    h = blake2b(token.encode("utf-8"), digest_size=32, person=salt).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _proj_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Xavier-ish scale so scores stay in a friendly range for demos
    return rng.standard_normal((rows, cols)).astype(np.float64) * np.sqrt(2.0 / (rows + cols))


@dataclass(frozen=True)
class AttentionResult:
    tokens: list[str]
    embeddings: np.ndarray  # (n, d_model)
    q: np.ndarray  # (n, d_k)
    k: np.ndarray  # (n, d_k)
    v: np.ndarray  # (n, d_k)
    scores: np.ndarray  # (n, n) pre-softmax
    weights: np.ndarray  # (n, n) post-softmax
    output: np.ndarray  # (n, d_k)
    d_model: int
    d_k: int

    def weights_list(self) -> list[list[float]]:
        return self.weights.tolist()

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "d_model": self.d_model,
            "d_k": self.d_k,
            "weights": self.weights_list(),
            "scores": self.scores.tolist(),
            "formula": "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V",
            "note": "Toy untrained projections for education only.",
        }


def compute_self_attention(
    text_or_tokens: str | Sequence[str],
    *,
    d_model: int = DEFAULT_D_MODEL,
    d_k: int = DEFAULT_D_K,
    seed: int = 42,
) -> AttentionResult:
    """Run educational single-head self-attention on a short sentence."""
    if isinstance(text_or_tokens, str):
        tokens = tokenize(text_or_tokens)
    else:
        tokens = [str(t) for t in text_or_tokens] or ["(empty)"]

    n = len(tokens)
    emb = np.stack([_token_vec(t, d_model, b"fastart-emb") for t in tokens], axis=0)

    W_q = _proj_matrix(d_model, d_k, seed + 1)
    W_k = _proj_matrix(d_model, d_k, seed + 2)
    W_v = _proj_matrix(d_model, d_k, seed + 3)

    q = emb @ W_q
    k = emb @ W_k
    v = emb @ W_v

    scale = np.sqrt(float(d_k))
    scores = (q @ k.T) / scale
    # numerically stable softmax per query row
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=1, keepdims=True)
    output = weights @ v

    return AttentionResult(
        tokens=tokens,
        embeddings=emb,
        q=q,
        k=k,
        v=v,
        scores=scores,
        weights=weights,
        output=output,
        d_model=d_model,
        d_k=d_k,
    )
