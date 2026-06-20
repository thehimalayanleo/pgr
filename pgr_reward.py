# pgr_reward.py
import numpy as np
from sentence_transformers import SentenceTransformer
from pgr_utils import segment_steps, omp_reconstruction_errors


class PGRReward:
    def __init__(
        self,
        dictionary_path: str = "dictionary_atoms.npy",
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        n_nonzero: int = 5,
        alpha: float = 0.5,        # weight on per-step reward
        temperature: float = 0.3,  # controls sharpness of exp(-error/τ)
        device: str = "cuda"
    ):
        self.D = np.load(dictionary_path)  # (k, d)
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.n_nonzero = n_nonzero
        self.alpha = alpha
        self.tau = temperature

    def segment_steps(self, text: str) -> list[str]:
        return segment_steps(text)

    def omp_error(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns per-step reconstruction error via OMP."""
        return omp_reconstruction_errors(self.D, embeddings, n_nonzero=self.n_nonzero)

    def step_rewards(self, errors: np.ndarray) -> np.ndarray:
        """Convert reconstruction errors to [0, 1] rewards."""
        return np.exp(-errors / self.tau)

    def __call__(
        self,
        prompt: str,
        completion: str,
        is_correct: bool | None = None  # None = oracle-free mode
    ) -> dict:
        steps = self.segment_steps(completion)

        if len(steps) == 0:
            return {"total": 0.0, "step_rewards": [], "n_steps": 0}

        embeddings = self.encoder.encode(
            steps,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        )

        errors = self.omp_error(embeddings)
        sr = self.step_rewards(errors)
        mean_step_reward = float(sr.mean())

        # Terminal reward
        if is_correct is None:
            terminal = 0.0
            total = mean_step_reward  # oracle-free
        else:
            terminal = 1.0 if is_correct else 0.0
            total = self.alpha * mean_step_reward + (1 - self.alpha) * terminal

        return {
            "total": total,
            "step_rewards": sr.tolist(),
            "terminal_reward": terminal,
            "mean_step_reward": mean_step_reward,
            "n_steps": len(steps),
            "reconstruction_errors": errors.tolist()
        }
