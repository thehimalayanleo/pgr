# modal_config.py
"""
Shared Modal infrastructure for PGR experiments.

Provides reusable image definitions and volume references so individual
scripts don't repeat the same pip_install lists.
"""

import modal

# ─────────────────────────────────────────────────────────────────────────
# Volume (persistent artifact storage)
# ─────────────────────────────────────────────────────────────────────────

volume = modal.Volume.from_name("pgr-artifacts", create_if_missing=True)
VOLUME_MOUNT = {"/artifacts": volume}

# ─────────────────────────────────────────────────────────────────────────
# Common pip packages grouped by role
# ─────────────────────────────────────────────────────────────────────────

_BASE_PACKAGES = [
    "torch==2.4.0",
    "numpy",
    "datasets",
]

_ENCODER_PACKAGES = [
    "sentence-transformers",
    "scikit-learn",
]

_INFERENCE_PACKAGES = [
    "transformers==4.46.2",
    "vllm==0.6.3",
]

_TRAINING_PACKAGES = [
    "transformers==4.46.2",
    "trl==0.14.0",
    "accelerate==0.34.2",
    "peft",
]

# ─────────────────────────────────────────────────────────────────────────
# Pre-built images
# ─────────────────────────────────────────────────────────────────────────

base_image = modal.Image.debian_slim(python_version="3.11")

image_encoder = base_image.pip_install(*_BASE_PACKAGES, *_ENCODER_PACKAGES)

image_inference = base_image.pip_install(*_BASE_PACKAGES, *_INFERENCE_PACKAGES)

image_training = base_image.pip_install(
    *_BASE_PACKAGES, *_ENCODER_PACKAGES, *_TRAINING_PACKAGES
)

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

ENCODER_NAME = "BAAI/bge-small-en-v1.5"
DICTIONARY_PATH = "/artifacts/dictionary_atoms.npy"
DATASET_NAME = "lighteval/MATH-Hard"
