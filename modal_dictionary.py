# modal_dictionary.py
"""
Offline dictionary learning phase.
Collects correct MATH solution steps, encodes them with a 22M sentence encoder,
and fits a sparse dictionary via DictionaryLearning (LARS).

Runtime: ~15 min on A10G
Cost:    ~$0.30
"""

import modal

app = modal.App("pgr-dictionary")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "sentence-transformers", "scikit-learn",
        "datasets", "numpy", "huggingface_hub"
    )
)

volume = modal.Volume.from_name("pgr-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/artifacts": volume}
)
def build_dictionary(n_atoms: int = 256, n_steps_sample: int = 10_000):
    import re
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import DictionaryLearning

    def segment_steps(text):
        steps = re.split(r'\n\n+|(?=Step \d+:)|(?=\d+\.)', text.strip())
        return [s.strip() for s in steps if len(s.strip()) > 20]

    # Load hard MATH problems
    ds = load_dataset("lighteval/MATH-Hard", split="train")
    hard = list(ds)

    all_steps = []
    for ex in hard:
        all_steps.extend(segment_steps(ex["solution"]))
    if not all_steps:
        raise RuntimeError(
            "No reasoning steps extracted from dataset. "
            "Check that the dataset contains solutions with multi-step reasoning."
        )
    print(f"Total steps collected: {len(all_steps)}")

    # Encode
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = encoder.encode(
        all_steps,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Sample and fit dictionary
    np.random.seed(42)
    if len(embeddings) < n_steps_sample:
        print(f"  Warning: only {len(embeddings)} embeddings available, "
              f"using all instead of sampling {n_steps_sample}")
        X = embeddings
    else:
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
