# modal_atom_sweep.py
"""
Atom count ablation: build dictionaries at {64, 128, 256, 512, 1024} atoms
and measure omp_gap on the shuffled-step probe.

Interpretation of the resulting curve:
  - monotonically decreasing in atom count → dictionary is overfitting
  - flat across atom counts → encoder is the bottleneck
  - peak in the middle → there's a Goldilocks zone

Runtime: ~20 min on A10G
Cost:    ~$0.40
"""

import modal
from modal_config import image_encoder, volume, VOLUME_MOUNT, DATASET_NAME

app = modal.App("pgr-atom-sweep")

ATOM_COUNTS = [64, 128, 256, 512, 1024]
N_PROBES    = 300   # number of (correct, shuffled) pairs to evaluate on


@app.function(
    image=image_encoder,
    gpu="A10G",
    timeout=3600,
    volumes=VOLUME_MOUNT,
)
def atom_sweep(encoder_name: str = "BAAI/bge-small-en-v1.5"):
    import json, random
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import DictionaryLearning
    from sklearn.metrics import roc_auc_score
    from pgr_utils import segment_steps, omp_reconstruction_errors

    def shuffle_words(step):
        words = step.split()
        random.shuffle(words)
        return " ".join(words)

    # ── Load + encode corpus ─────────────────────────────────────────────
    print("Loading MATH-Hard train split…")
    ds = load_dataset(DATASET_NAME, split="train")
    all_steps = []
    for ex in ds:
        all_steps.extend(segment_steps(ex["solution"]))
    print(f"Collected {len(all_steps)} reasoning steps")

    print(f"Loading encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name)
    print("Encoding all steps…")
    X = encoder.encode(all_steps, normalize_embeddings=True, batch_size=256,
                       show_progress_bar=True)
    print(f"Embeddings: {X.shape}")

    # ── Build probe set from test split ──────────────────────────────────
    print("\nBuilding probe set from test split…")
    test_ds = load_dataset(DATASET_NAME, split="test")
    random.seed(42)
    np.random.seed(42)

    probe_correct, probe_shuffled = [], []
    for ex in test_ds:
        steps = segment_steps(ex["solution"])
        for s in steps:
            probe_correct.append(s)
            probe_shuffled.append(shuffle_words(s))
            if len(probe_correct) >= N_PROBES:
                break
        if len(probe_correct) >= N_PROBES:
            break

    print(f"Probe set: {len(probe_correct)} correct, {len(probe_shuffled)} shuffled")
    emb_correct  = encoder.encode(probe_correct,  normalize_embeddings=True, batch_size=128)
    emb_shuffled = encoder.encode(probe_shuffled, normalize_embeddings=True, batch_size=128)

    # ── Sweep over atom counts ───────────────────────────────────────────
    results = []

    # Sample subset for dictionary fitting (faster, same as smoke test)
    np.random.seed(42)
    n_sample = min(10_000, len(X))
    idx = np.random.choice(len(X), size=n_sample, replace=False)
    X_train = X[idx]
    print(f"\nDictionary fit corpus: {n_sample} steps")

    for n_atoms in ATOM_COUNTS:
        print(f"\n{'='*55}")
        print(f"  ATOMS = {n_atoms}")
        print(f"{'='*55}")

        dl = DictionaryLearning(
            n_components=n_atoms,
            alpha=0.5,
            max_iter=300,
            fit_algorithm="lars",
            transform_algorithm="omp",
            transform_n_nonzero_coefs=5,
            n_jobs=-1,
        )
        dl.fit(X_train)
        D = dl.components_   # shape: (n_atoms, 384)

        err_correct  = omp_reconstruction_errors(D, emb_correct,  n_nonzero=5)
        err_shuffled = omp_reconstruction_errors(D, emb_shuffled, n_nonzero=5)

        mean_correct  = float(err_correct.mean())
        mean_shuffled = float(err_shuffled.mean())
        gap           = mean_shuffled - mean_correct

        # AUROC: can OMP error rank correct < shuffled?
        labels  = [1]*len(err_correct) + [0]*len(err_shuffled)
        # Higher score = less error = more likely "correct"
        scores  = list(-err_correct) + list(-err_shuffled)
        auroc   = float(roc_auc_score(labels, scores))

        row = {
            "n_atoms": n_atoms,
            "mean_err_correct": round(mean_correct, 4),
            "mean_err_shuffled": round(mean_shuffled, 4),
            "omp_gap": round(gap, 4),
            "auroc": round(auroc, 4),
        }
        results.append(row)
        print(f"  correct_err = {mean_correct:.4f}")
        print(f"  shuffled_err = {mean_shuffled:.4f}")
        print(f"  gap = {gap:.4f}")
        print(f"  AUROC = {auroc:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  ATOM COUNT SWEEP — SUMMARY")
    print(f"{'='*55}")
    print(f"{'atoms':>8} {'correct':>10} {'shuffled':>10} {'gap':>10} {'AUROC':>10}")
    for r in results:
        print(f"{r['n_atoms']:>8} {r['mean_err_correct']:>10} {r['mean_err_shuffled']:>10} "
              f"{r['omp_gap']:>10} {r['auroc']:>10}")

    # Interpretation
    print(f"\n{'='*55}")
    print("  INTERPRETATION")
    print(f"{'='*55}")
    gaps = [r["omp_gap"] for r in results]
    aurocs = [r["auroc"] for r in results]

    if all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1)):
        print("  Gap monotonically decreasing -> DICTIONARY OVERFITTING")
        print("     More atoms reduce discriminability. Use fewer atoms or sparser codes.")
    elif max(aurocs) - min(aurocs) < 0.02:
        print("  AUROC flat across atom counts -> ENCODER BOTTLENECK")
        print("     bge-small can't separate good vs bad reasoning. Try bge-large.")
    else:
        best = max(results, key=lambda r: r["auroc"])
        print(f"  Sweet spot found at {best['n_atoms']} atoms (AUROC={best['auroc']:.4f})")
        print(f"     Goldilocks zone exists. Use {best['n_atoms']}-atom dictionary.")

    # Save results — one file per encoder
    encoder_slug = encoder_name.replace("/", "_")
    out_path = f"/artifacts/atom_sweep_{encoder_slug}.json"
    with open(out_path, "w") as f:
        json.dump({"encoder": encoder_name, "results": results}, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_path}")

    return results


@app.local_entrypoint()
def main(encoder: str = "BAAI/bge-large-en-v1.5"):
    print(f"Running atom sweep with encoder: {encoder}")
    results = atom_sweep.remote(encoder_name=encoder)
    print("\nFinal results:")
    for r in results:
        print(r)
