"""Unit tests for step_pgr_trainer.py — utility functions and advantage computation."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from step_pgr_trainer import (
    segment_steps,
    extract_answer,
    omp_step_rewards,
    step_token_spans,
    validate_regression,
)


# ---------------------------------------------------------------------------
# Test segment_steps
# ---------------------------------------------------------------------------

class TestSegmentSteps:
    """Tests for the standalone segment_steps function."""

    def test_splits_on_double_newline(self):
        text = "This is a reasonably long first step.\n\nThis is a reasonably long second step."
        steps = segment_steps(text)
        assert len(steps) == 2

    def test_splits_on_step_prefix(self):
        text = "Step 1: Solve the equation by isolating x.\nStep 2: Divide both sides by the coefficient."
        steps = segment_steps(text)
        assert len(steps) == 2

    def test_filters_short_segments(self):
        text = "Hi.\n\nThis is a step that is long enough to pass the filter."
        steps = segment_steps(text)
        assert len(steps) == 1

    def test_empty_string(self):
        assert segment_steps("") == []

    def test_whitespace_only(self):
        assert segment_steps("   \n\n  ") == []

    def test_no_delimiters_single_long_string(self):
        text = "A single continuous reasoning passage with no step delimiters at all"
        steps = segment_steps(text)
        assert len(steps) == 1

    def test_multiple_double_newlines(self):
        text = "First step is over twenty chars.\n\n\n\nSecond step is over twenty chars."
        steps = segment_steps(text)
        assert len(steps) == 2


# ---------------------------------------------------------------------------
# Test extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    """Tests for the extract_answer function (LaTeX \\boxed{} extraction)."""

    def test_simple_boxed(self):
        assert extract_answer("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        result = extract_answer("\\boxed{\\frac{1}{2}}")
        assert result == "\\frac{1}{2}"

    def test_deeply_nested(self):
        result = extract_answer("\\boxed{\\sqrt{\\frac{a}{b}}}")
        assert result == "\\sqrt{\\frac{a}{b}}"

    def test_no_boxed(self):
        assert extract_answer("The answer is 42") is None

    def test_empty_boxed(self):
        result = extract_answer("\\boxed{}")
        assert result == ""

    def test_boxed_with_spaces(self):
        result = extract_answer("\\boxed{  x + 1  }")
        assert result == "x + 1"

    def test_multiple_boxed_returns_first(self):
        result = extract_answer("\\boxed{first} and \\boxed{second}")
        assert result == "first"

    def test_unclosed_boxed_returns_none(self):
        result = extract_answer("\\boxed{unclosed")
        assert result is None

    def test_boxed_with_complex_latex(self):
        result = extract_answer("\\boxed{\\frac{\\sqrt{3}}{2}}")
        assert result == "\\frac{\\sqrt{3}}{2}"

    def test_text_before_and_after_boxed(self):
        text = "After simplification, we get \\boxed{x^2 + 1}. Done."
        assert extract_answer(text) == "x^2 + 1"


# ---------------------------------------------------------------------------
# Test omp_step_rewards
# ---------------------------------------------------------------------------

class TestOmpStepRewards:
    """Tests for omp_step_rewards function."""

    def test_empty_steps_returns_zero(self):
        D = np.random.randn(32, 64)
        encoder = MagicMock()
        result = omp_step_rewards([], encoder, D)
        np.testing.assert_array_equal(result, np.array([0.0]))

    def test_output_shape(self):
        k, d = 32, 64
        D = np.random.randn(k, d)
        D /= np.linalg.norm(D, axis=1, keepdims=True)

        mock_encoder = MagicMock()
        steps = ["step one is long enough here", "step two is also long enough"]
        mock_encoder.encode = MagicMock(
            return_value=np.random.randn(len(steps), d).astype(np.float32)
        )
        result = omp_step_rewards(steps, mock_encoder, D)
        assert result.shape == (len(steps),)

    def test_output_in_zero_one(self):
        k, d = 32, 64
        D = np.random.randn(k, d)
        D /= np.linalg.norm(D, axis=1, keepdims=True)

        mock_encoder = MagicMock()
        steps = ["reasoning step about mathematics"]
        mock_encoder.encode = MagicMock(
            return_value=np.random.randn(1, d).astype(np.float32)
        )
        result = omp_step_rewards(steps, mock_encoder, D, tau=0.3)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_in_span_embedding_gets_higher_reward(self):
        k, d = 32, 64
        D = np.random.randn(k, d)
        D /= np.linalg.norm(D, axis=1, keepdims=True)

        # Embedding in span of dictionary atoms → lower error → higher reward
        coeffs = np.zeros(k)
        coeffs[:3] = [0.5, 0.3, 0.2]
        in_span = (coeffs @ D).reshape(1, -1)

        random_emb = np.random.randn(1, d) * 5

        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(return_value=in_span)
        r_in_span = omp_step_rewards(["step in span"], mock_encoder, D)

        mock_encoder.encode = MagicMock(return_value=random_emb)
        r_random = omp_step_rewards(["random step"], mock_encoder, D)

        assert r_in_span[0] > r_random[0]

    def test_tau_parameter_affects_rewards(self):
        k, d = 32, 64
        D = np.random.randn(k, d)
        D /= np.linalg.norm(D, axis=1, keepdims=True)

        mock_encoder = MagicMock()
        emb = np.random.randn(1, d).astype(np.float32)
        mock_encoder.encode = MagicMock(return_value=emb)

        r_low = omp_step_rewards(["step"], mock_encoder, D, tau=0.1)
        mock_encoder.encode = MagicMock(return_value=emb)
        r_high = omp_step_rewards(["step"], mock_encoder, D, tau=1.0)
        assert r_high[0] >= r_low[0]


# ---------------------------------------------------------------------------
# Test step_token_spans
# ---------------------------------------------------------------------------

class TestStepTokenSpans:
    """Tests for step_token_spans — maps reasoning steps to token indices."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that uses offset_mapping."""
        tok = MagicMock()

        def call_fn(text, return_offsets_mapping=False, add_special_tokens=False, **kwargs):
            offsets = []
            pos = 0
            for word in text.split():
                start = text.find(word, pos)
                end = start + len(word)
                offsets.append((start, end))
                pos = end
            result = {"input_ids": list(range(len(offsets)))}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            return result

        tok.__call__ = call_fn
        tok.side_effect = call_fn

        def encode_fn(text, add_special_tokens=False):
            return list(range(len(text.split())))

        tok.encode = encode_fn
        return tok

    def test_empty_text(self, mock_tokenizer):
        result = step_token_spans(mock_tokenizer, "", [])
        assert result == []

    def test_short_text_no_steps(self, mock_tokenizer):
        result = step_token_spans(mock_tokenizer, "Hi there.", [0, 1])
        assert result == []

    def test_returns_tuples_with_correct_structure(self, mock_tokenizer):
        text = "This is a long enough first step here.\n\nThis is a long enough second step here."
        token_ids = list(range(len(text.split())))
        result = step_token_spans(mock_tokenizer, text, token_ids)
        for span in result:
            assert len(span) == 3
            start, end, step_text = span
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(step_text, str)
            assert end > start

    def test_token_spans_non_overlapping(self, mock_tokenizer):
        text = "This is a long enough first step here.\n\nThis is a long enough second step here."
        token_ids = list(range(len(text.split())))
        spans = step_token_spans(mock_tokenizer, text, token_ids)
        for i in range(len(spans) - 1):
            _, end_i, _ = spans[i]
            start_next, _, _ = spans[i + 1]
            assert end_i <= start_next, "Token spans should not overlap"


# ---------------------------------------------------------------------------
# Test _compute_step_advantages (pooled mode)
# ---------------------------------------------------------------------------

class TestComputeStepAdvantages:
    """Tests for the pooled advantage normalization logic."""

    def _compute_pooled(self, all_step_rewards, num_generations):
        """Reimplementation of the pooled advantage logic for testing."""
        n_rollouts = len(all_step_rewards)
        groups = n_rollouts // num_generations
        out = [None] * n_rollouts

        for g in range(groups):
            slc = slice(g * num_generations, (g + 1) * num_generations)
            group_rewards = all_step_rewards[slc]
            pooled = np.concatenate(group_rewards) if any(len(r) for r in group_rewards) else np.array([0.0])
            mu, sigma = pooled.mean(), pooled.std()
            for i in range(num_generations):
                out[g * num_generations + i] = (group_rewards[i] - mu) / (sigma + 1e-4)
        return out

    def test_uniform_rewards_zero_advantage(self):
        rewards = [np.full(3, 0.5), np.full(4, 0.5), np.full(3, 0.5), np.full(4, 0.5)]
        adv = self._compute_pooled(rewards, num_generations=4)
        for a in adv:
            np.testing.assert_allclose(a, 0.0, atol=1e-2)

    def test_better_rollout_positive_advantage(self):
        rewards = [
            np.full(3, 0.9),
            np.full(3, 0.1),
            np.full(3, 0.5),
            np.full(3, 0.5),
        ]
        adv = self._compute_pooled(rewards, num_generations=4)
        assert np.all(adv[0] > 0)
        assert np.all(adv[1] < 0)

    def test_output_count_matches_input(self):
        rewards = [np.array([0.3, 0.4]), np.array([0.6, 0.7])]
        adv = self._compute_pooled(rewards, num_generations=2)
        assert len(adv) == 2
        assert len(adv[0]) == 2
        assert len(adv[1]) == 2

    def test_multiple_groups(self):
        rewards = [
            np.array([0.9, 0.9]),
            np.array([0.1, 0.1]),
            np.array([0.8, 0.8]),
            np.array([0.2, 0.2]),
        ]
        adv = self._compute_pooled(rewards, num_generations=2)
        assert np.all(adv[0] > 0)
        assert np.all(adv[1] < 0)
        assert np.all(adv[2] > 0)
        assert np.all(adv[3] < 0)

    def test_constant_rewards_per_rollout_yields_uniform_tokens(self):
        """Regression: constant step rewards -> uniform per-token advantages."""
        rewards = [
            np.full(5, 0.3),
            np.full(5, 0.7),
            np.full(5, 0.1),
            np.full(5, 0.5),
        ]
        adv = self._compute_pooled(rewards, num_generations=4)
        for i, a in enumerate(adv):
            assert np.allclose(a, a[0]), \
                f"Rollout {i}: constant step rewards should give uniform advantage"

    def test_advantages_are_zero_mean_across_group(self):
        rewards = [
            np.array([0.2, 0.8, 0.5]),
            np.array([0.9, 0.3]),
            np.array([0.6, 0.4, 0.7, 0.1]),
            np.array([0.5, 0.5]),
        ]
        adv = self._compute_pooled(rewards, num_generations=4)
        all_values = np.concatenate(adv)
        assert abs(all_values.mean()) < 0.01


# ---------------------------------------------------------------------------
# Test _compute_step_advantages (group_mean mode)
# ---------------------------------------------------------------------------

class TestComputeStepAdvantagesGroupMean:
    """Tests for the group_mean advantage mode."""

    def _compute_group_mean(self, all_step_rewards, num_generations):
        n_rollouts = len(all_step_rewards)
        groups = n_rollouts // num_generations
        out = [None] * n_rollouts

        for g in range(groups):
            slc = slice(g * num_generations, (g + 1) * num_generations)
            group_rewards = all_step_rewards[slc]
            group_trajectory_means = np.array([r.mean() if len(r) else 0.0 for r in group_rewards])
            mu_traj = group_trajectory_means.mean()
            sigma_traj = group_trajectory_means.std()
            for i in range(num_generations):
                out[g * num_generations + i] = (group_rewards[i] - mu_traj) / (sigma_traj + 1e-4)
        return out

    def test_better_trajectory_positive(self):
        rewards = [
            np.array([0.8, 0.9]),
            np.array([0.2, 0.1]),
        ]
        adv = self._compute_group_mean(rewards, num_generations=2)
        assert np.all(adv[0] > 0)
        assert np.all(adv[1] < 0)

    def test_uniform_means_zero_advantage(self):
        rewards = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
        ]
        adv = self._compute_group_mean(rewards, num_generations=2)
        for a in adv:
            np.testing.assert_allclose(a, 0.0, atol=1e-2)

    def test_within_rollout_variation_preserved(self):
        rewards = [
            np.array([0.2, 0.8]),
            np.array([0.5, 0.5]),
        ]
        adv = self._compute_group_mean(rewards, num_generations=2)
        assert adv[0][0] != adv[0][1]
        np.testing.assert_allclose(adv[1][0], adv[1][1])


# ---------------------------------------------------------------------------
# Test validate_regression (integration-style)
# ---------------------------------------------------------------------------

class TestValidateRegression:
    """Test the validate_regression function from step_pgr_trainer.py."""

    def test_runs_without_error(self, capsys):
        validate_regression()
        captured = capsys.readouterr()
        assert "regression test passed" in captured.out
