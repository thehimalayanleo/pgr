"""Unit tests for pgr_reward.py — PGRReward class and helpers."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import pgr_reward
from pgr_reward import PGRReward


# ---------------------------------------------------------------------------
# Helper: build a PGRReward with mocked encoder and small dictionary
# ---------------------------------------------------------------------------

def _make_reward(k=32, d=64, alpha=0.5, tau=0.3):
    """Build PGRReward without loading files or downloading models."""
    D = np.random.randn(k, d).astype(np.float64)
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    mock_encoder = MagicMock()
    mock_encoder.encode = MagicMock(
        side_effect=lambda texts, **kwargs: np.random.randn(len(texts), d).astype(np.float32)
    )

    with patch.object(pgr_reward.np, "load", return_value=D), \
         patch.object(pgr_reward, "SentenceTransformer", return_value=mock_encoder):
        r = PGRReward(
            dictionary_path="fake.npy",
            device="cpu",
            alpha=alpha,
            temperature=tau,
        )
    return r


# ---------------------------------------------------------------------------
# Test segment_steps (pure string logic, no dependencies)
# ---------------------------------------------------------------------------

class TestSegmentSteps:
    """Tests for PGRReward.segment_steps."""

    @pytest.fixture
    def reward(self):
        return _make_reward()

    def test_splits_on_double_newline(self, reward):
        text = "This is a long first step here.\n\nThis is a long second step here."
        steps = reward.segment_steps(text)
        assert len(steps) == 2
        assert steps[0] == "This is a long first step here."
        assert steps[1] == "This is a long second step here."

    def test_splits_on_step_prefix(self, reward):
        text = "Step 1: Solve the equation by isolating x.\nStep 2: Divide both sides by the coefficient."
        steps = reward.segment_steps(text)
        assert len(steps) == 2

    def test_splits_on_numbered_prefix(self, reward):
        text = "1. Set up the equation 2x + 5 = 13.\n2. Subtract 5 from both sides gives 2x = 8."
        steps = reward.segment_steps(text)
        assert len(steps) == 2

    def test_filters_short_segments(self, reward):
        text = "Short.\n\nThis is a sufficiently long step that exceeds 20 chars."
        steps = reward.segment_steps(text)
        assert len(steps) == 1
        assert "sufficiently long" in steps[0]

    def test_empty_input(self, reward):
        assert reward.segment_steps("") == []
        assert reward.segment_steps("   ") == []

    def test_single_long_step(self, reward):
        text = "This is a single long reasoning step with no delimiters."
        steps = reward.segment_steps(text)
        assert len(steps) == 1

    def test_many_steps(self, reward):
        text = "\n\n".join([f"Step number {i}: this is reasoning about the problem" for i in range(10)])
        steps = reward.segment_steps(text)
        assert len(steps) == 10


# ---------------------------------------------------------------------------
# Test omp_error (numerical computation)
# ---------------------------------------------------------------------------

class TestOmpError:
    """Tests for PGRReward.omp_error."""

    @pytest.fixture
    def reward(self):
        return _make_reward(k=32, d=64)

    def test_output_shape(self, reward):
        n_steps, d = 5, reward.D.shape[1]
        embeddings = np.random.randn(n_steps, d)
        errors = reward.omp_error(embeddings)
        assert errors.shape == (n_steps,)

    def test_errors_non_negative(self, reward):
        n_steps, d = 10, reward.D.shape[1]
        embeddings = np.random.randn(n_steps, d)
        errors = reward.omp_error(embeddings)
        assert np.all(errors >= 0)

    def test_in_span_embedding_lower_error(self, reward):
        """An embedding in the span of dictionary atoms has lower error than random."""
        D = reward.D
        # Create an embedding as a linear combination of dictionary atoms
        coeffs = np.zeros(D.shape[0])
        coeffs[:3] = [0.5, 0.3, 0.2]
        in_span = coeffs @ D  # linear combo of first 3 atoms
        in_span = in_span.reshape(1, -1)
        
        random_emb = np.random.randn(1, D.shape[1]) * 5
        
        error_in_span = reward.omp_error(in_span)
        error_random = reward.omp_error(random_emb)
        assert error_in_span[0] < error_random[0]

    def test_multiple_steps(self, reward):
        """OMP error with multiple steps returns correct shape."""
        d = reward.D.shape[1]
        embeddings = np.random.randn(3, d)
        errors = reward.omp_error(embeddings)
        assert errors.shape == (3,)
        assert np.all(errors >= 0)

    def test_random_embedding_has_positive_error(self, reward):
        """A random embedding (not in dictionary span) should have nonzero error."""
        d = reward.D.shape[1]
        embeddings = np.random.randn(1, d) * 10
        errors = reward.omp_error(embeddings)
        assert errors[0] > 0


# ---------------------------------------------------------------------------
# Test step_rewards (simple exponential transform)
# ---------------------------------------------------------------------------

class TestStepRewards:

    @pytest.fixture
    def reward(self):
        return _make_reward()

    def test_zero_error_gives_reward_one(self, reward):
        errors = np.array([0.0, 0.0, 0.0])
        rewards = reward.step_rewards(errors)
        np.testing.assert_allclose(rewards, 1.0)

    def test_high_error_gives_low_reward(self, reward):
        errors = np.array([10.0, 20.0])
        rewards = reward.step_rewards(errors)
        assert np.all(rewards < 0.01)

    def test_output_in_zero_one(self, reward):
        errors = np.random.uniform(0, 5, size=20)
        rewards = reward.step_rewards(errors)
        assert np.all(rewards >= 0)
        assert np.all(rewards <= 1)

    def test_monotonically_decreasing(self, reward):
        errors = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        rewards = reward.step_rewards(errors)
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1]

    def test_temperature_effect(self):
        """Higher tau -> softer decay (higher rewards for same error)."""
        r_low_tau = _make_reward(tau=0.1)
        r_high_tau = _make_reward(tau=1.0)
        errors = np.array([1.0])
        assert r_high_tau.step_rewards(errors)[0] > r_low_tau.step_rewards(errors)[0]


# ---------------------------------------------------------------------------
# Test __call__ (full pipeline with mocked encoder)
# ---------------------------------------------------------------------------

class TestPGRRewardCall:

    @pytest.fixture
    def reward(self):
        return _make_reward(k=32, d=64, alpha=0.5, tau=0.3)

    def test_empty_completion(self, reward):
        result = reward("What is 2+2?", "")
        assert result["total"] == 0.0
        assert result["step_rewards"] == []
        assert result["n_steps"] == 0

    def test_oracle_free_mode(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("What is 2+2?", completion, is_correct=None)
        assert "total" in result
        assert "terminal_reward" in result
        assert result["terminal_reward"] == 0.0
        assert abs(result["total"] - result["mean_step_reward"]) < 1e-6

    def test_correct_answer_mode(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("What is 2+2?", completion, is_correct=True)
        assert result["terminal_reward"] == 1.0
        expected_total = 0.5 * result["mean_step_reward"] + 0.5 * 1.0
        assert abs(result["total"] - expected_total) < 1e-6

    def test_incorrect_answer_mode(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("What is 2+2?", completion, is_correct=False)
        assert result["terminal_reward"] == 0.0
        expected_total = 0.5 * result["mean_step_reward"] + 0.5 * 0.0
        assert abs(result["total"] - expected_total) < 1e-6

    def test_output_keys(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("What is 2+2?", completion, is_correct=True)
        expected_keys = {"total", "step_rewards", "terminal_reward", "mean_step_reward", "n_steps", "reconstruction_errors"}
        assert set(result.keys()) == expected_keys

    def test_n_steps_matches_step_rewards_length(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("What is 2+2?", completion, is_correct=True)
        assert result["n_steps"] == len(result["step_rewards"])
        assert result["n_steps"] == len(result["reconstruction_errors"])

    def test_total_in_zero_one_for_correct(self, reward):
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = reward("prompt", completion, is_correct=True)
        assert 0.0 <= result["total"] <= 1.0

    def test_alpha_zero_gives_pure_terminal(self):
        """alpha=0 -> total = 0*step + 1*terminal."""
        r = _make_reward(alpha=0.0)
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = r("p", completion, is_correct=True)
        assert abs(result["total"] - 1.0) < 1e-6

    def test_alpha_one_gives_pure_step(self):
        """alpha=1 -> total = 1*step + 0*terminal = mean_step_reward."""
        r = _make_reward(alpha=1.0)
        completion = "Step 1: We first identify that x = 2.\n\nStep 2: Then we verify 2 + 2 = 4."
        result = r("p", completion, is_correct=True)
        assert abs(result["total"] - result["mean_step_reward"]) < 1e-6
