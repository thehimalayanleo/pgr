# modal_dictionary.py
"""
Offline dictionary learning phase.
Collects correct MATH solution steps, encodes them with a 22M sentence encoder,
and fits a sparse dictionary via DictionaryLearning (LARS).

Runtime: ~15 min on A10G
Cost:    ~$0.30
"""

import modal
from modal_config import image_encoder, volume, VOLUME_MOUNT, ENCODER_NAME, DATASET_NAME

app = modal.App("pgr-dictionary")


@app.function(
    image=image_encoder,
    gpu="A10G",
    timeout=3600,
    volumes=VOLUME_MOUNT,
)
def build_dictionary(n_atoms: int = 256, n_steps_sample: int = 10_000):
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import DictionaryLearning
    from pgr_utils import segment_steps

    ds = load_dataset(DATASET_NAME, split="train")
    hard = list(ds)

    all_steps = []
    for ex in hard:
        all_steps.extend(segment_steps(ex["solution"]))
    print(f"Total steps collected: {len(all_steps)}")

    encoder = SentenceTransformer(ENCODER_NAME)
    embeddings = encoder.encode(
        all_steps,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Sample and fit dictionary
    np.random.seed(42)
    idx = np.random.choice(len(embeddings), size=n_steps_sample, replace=False)
    X = embeddings[idx]

    dl = DictionaryLearning(
        n_components=n_atoms,
        alpha=0.5,
        max_iter=500,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=5,
        n_jobs=-1,
        verbose=1
    )
    dl.fit(X)

    np.save("/artifacts/dictionary_atoms.npy", dl.components_)
    np.save("/artifacts/all_embeddings_sample.npy", embeddings[:n_steps_sample])
    volume.commit()

    print(f"Saved dictionary: {dl.components_.shape}  (atoms × embedding_dim)")
    return {"atoms": n_atoms, "embedding_dim": dl.components_.shape[1]}


@app.local_entrypoint()
def main():
    result = build_dictionary.remote()
    print(result)
