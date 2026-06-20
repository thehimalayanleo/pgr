# pgr_utils.py
"""
Shared utilities for the PGR codebase.

Consolidates duplicated logic for:
  - Step segmentation from solution text
  - OMP reconstruction error and step-level rewards
  - LaTeX \\boxed{} answer extraction
  - Reproducible seeding
"""

import re
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Step segmentation
# ─────────────────────────────────────────────────────────────────────────

def segment_steps(text: str, min_length: int = 20) -> list[str]:
    """Split a solution trace into reasoning steps.

    Splits on double newlines, 'Step N:', or numbered list items ('N.').
    Filters out fragments shorter than `min_length` characters.
    """
    parts = re.split(r'\n\n+|(?=Step \d+:)|(?=\d+\.)', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > min_length]


# ─────────────────────────────────────────────────────────────────────────
# OMP reconstruction
# ─────────────────────────────────────────────────────────────────────────

def omp_reconstruction_errors(
    D: np.ndarray,
    embeddings: np.ndarray,
    n_nonzero: int = 5,
) -> np.ndarray:
    """Compute per-vector OMP reconstruction error against dictionary D.

    Args:
        D: Dictionary matrix, shape (n_atoms, embed_dim).
        embeddings: Input vectors, shape (n_samples, embed_dim).
        n_nonzero: Number of nonzero coefficients in OMP.

    Returns:
        Array of shape (n_samples,) with L2 reconstruction errors.
    """
    from sklearn.linear_model import orthogonal_mp

    codes = orthogonal_mp(D.T, embeddings.T, n_nonzero_coefs=n_nonzero)
    reconstructed = D.T @ codes
    return np.linalg.norm(embeddings.T - reconstructed, axis=0)


def omp_step_rewards(
    step_texts: list[str],
    encoder,
    D: np.ndarray,
    tau: float = 0.3,
    n_nonzero: int = 5,
) -> np.ndarray:
    """Encode steps and compute OMP reconstruction rewards in [0, 1].

    Returns exp(-error / tau) for each step. If step_texts is empty,
    returns np.array([0.0]).
    """
    if not step_texts:
        return np.array([0.0])
    emb = encoder.encode(step_texts, normalize_embeddings=True, batch_size=64)
    errors = omp_reconstruction_errors(D, emb, n_nonzero=n_nonzero)
    return np.exp(-errors / tau)


# ─────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of \\boxed{...} handling nested braces correctly.

    Returns None if no \\boxed{} is found or braces are unbalanced.
    """
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth > 0:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


# ─────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
