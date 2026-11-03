"""Tests for the sequence objectives and the CSPO band calibration."""

import numpy as np

from cspolab import objectives as obj


def test_sequence_scores_hand_computed():
    # Two responses, three response tokens each.
    logp = np.array([[-0.1, -0.2, -0.3, 0.0],
                     [-1.0, -0.5, -0.25, 0.0]])
    logp_old = np.array([[-0.2, -0.2, -0.2, 0.0],
                         [-0.5, -0.5, -0.5, 0.0]])
    s, w = obj.sequence_scores(logp, logp_old, [3, 3])
    # S_0 = 0.1 + 0 + (-0.1) = 0.0 ; S_1 = -0.5 + 0 + 0.25 = -0.25
    assert abs(s[0]) < 1e-12
    assert abs(s[1] - (-0.25)) < 1e-12
    assert abs(w[0] - 1.0) < 1e-12
    assert abs(w[1] - np.exp(-0.25 / 3)) < 1e-12


def test_group_advantages_hand_computed():
    rewards = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=float)  # G=4, 2 groups
    a = obj.group_advantages(rewards, 4)
    r = rewards.reshape(2, 4)
    expected = ((r - r.mean(axis=1, keepdims=True))
                / (r.std(axis=1, keepdims=True) + 1e-8)).ravel()
    assert np.allclose(a, expected)
    # Zero-variance group -> all-zero advantages.
    assert np.all(obj.group_advantages(np.zeros(4), 4) == 0.0)
    # Advantages sum to ~0 within each group.
    assert abs(a[:4].sum()) < 1e-9


def _brute_gspo(w, advantages, a, b):
    return float(np.mean([min(wi * ai, min(max(wi, a), b) * ai)
                          for wi, ai in zip(w, advantages)]))


def test_gspo_objective_vs_brute_force():
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.6, size=16)
    adv = rng.normal(size=16)
    for (a, b) in [(0.8, 1.2), (0.9, 1.1), (0.0, np.inf)]:
        assert abs(obj.sequence_objective(w, adv, a, b)
                   - _brute_gspo(w, adv, a, b)) < 1e-12


def _brute_grpo(ratios, advantages, eps, lengths, mask):
    a, b = 1 - eps, 1 + eps
    vals = []
    for i in range(len(advantages)):
        toks = [min(r * advantages[i], min(max(r, a), b) * advantages[i])
                for r, m in zip(ratios[i], mask[i]) if m]
        vals.append(np.mean(toks))
    return float(np.mean(vals))


def test_grpo_objective_vs_brute_force():
    rng = np.random.default_rng(1)
    ratios = rng.uniform(0.6, 1.5, size=(8, 6))
    adv = rng.normal(size=8)
    lengths = np.array([4, 6, 3, 6, 5, 2, 6, 4])
    mask = np.zeros((8, 6), dtype=bool)
    for i, ln in enumerate(lengths):
        mask[i, :ln] = True
    for eps in (0.2, 0.1):
        got = obj.grpo_objective(ratios, adv, eps, lengths, mask)
        want = _brute_grpo(ratios, adv, eps, lengths, mask)
        assert abs(got - want) < 1e-12


def test_fspo_band_endpoints():
    lengths = np.array([4, 9, 16, 25])
    a, b = obj.fspo_band(lengths, c=0.2)
    assert np.allclose(a, np.exp(-0.2 / np.sqrt(lengths)))
    assert np.allclose(b, np.exp(0.2 / np.sqrt(lengths)))
    # Longer sequences get narrower bands in W (fairness).
    assert b[0] > b[1] > b[2] > b[3] > 1.0
    assert a[0] < a[1] < a[2] < a[3] < 1.0


def test_calibrate_band_quantiles_and_anchoring():
    buf = np.array([0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3])
    a, b = obj.calibrate_band(buf, alpha=0.2)
    # alpha/2 = 0.1 quantile (lower) of 9 values -> smallest order stat
    # at/above the 10th percentile; 1-0.1 -> near the top.
    assert a == min(float(np.quantile(buf, 0.1, method="lower")), 1.0)
    assert b == max(float(np.quantile(buf, 0.9, method="lower")), 1.0)
    assert a <= 1.0 <= b


def test_calibrate_band_warm_start_on_empty_buffer():
    a, b = obj.calibrate_band(np.array([]), alpha=0.1, warm_c=0.2)
    assert (a, b) == (0.8, 1.2)
    a, b = obj.calibrate_band(None, alpha=0.1, warm_c=0.2)
    assert (a, b) == (0.8, 1.2)


def test_calibrate_band_all_scores_above_one():
    # Anchoring: if every retained score is > 1, the lower endpoint
    # is still 1, not the smallest score.
    buf = np.array([1.05, 1.1, 1.2, 1.3])
    a, b = obj.calibrate_band(buf, alpha=0.1)
    assert a == 1.0
    assert b > 1.0


def test_rolling_buffer_keeps_last_k_blocks():
    buf = obj.RollingBuffer(k=2)
    buf.add_block([1.0, 1.0])
    buf.add_block([0.9, 1.1])
    assert len(buf) == 4
    buf.add_block([0.8, 1.2])
    assert len(buf) == 4  # first block dropped
    assert set(buf.values()) == {0.9, 1.1, 0.8, 1.2}


def test_miscoverage_on_exchangeable_scores_near_alpha():
    rng = np.random.default_rng(0)
    buffer = rng.normal(1.0, 0.15, size=2000)
    fresh = rng.normal(1.0, 0.15, size=2000)
    a, b = obj.calibrate_band(buffer, alpha=0.1)
    misc = obj.miscoverage(fresh, a, b)
    # Exchangeable scores: realised miscoverage concentrates near
    # alpha (binomial sd ~0.007 at n=2000); allow a wide margin.
    assert 0.04 <= misc <= 0.16


def test_prop33_bound_holds_on_synthetic_draws():
    rng = np.random.default_rng(2)
    # W uniform in [0.8, 1.25]: E[W^2] ~ 1.07 <= 1 + 2/L_min with L_min 6.
    w = rng.uniform(0.8, 1.25, size=64)
    assert np.mean(w ** 2) <= 1.0 + 2.0 / 6.0
    rewards = rng.integers(0, 2, size=64).astype(float)
    adv = obj.group_advantages(rewards, 8)
    out = obj.prop33_check(w, adv, 0.95, 1.05, l_min=6, group_size=8)
    assert out["gap"] >= -1e-12
    assert out["holds"]


def test_lemma32_bound_on_toy_policy():
    rng = np.random.default_rng(3)
    # Behaviour uniform over 2 tokens; target [0.6, 0.4]. Sample
    # sequences of length 8, W = exp(mean log ratio).
    length = 8
    n = 4000
    draws = rng.integers(0, 2, size=(n, length))
    ratios = np.where(draws == 0, 1.2, 0.8)
    s = np.log(ratios).sum(axis=1)
    w = np.exp(s / length)
    out = obj.lemma32_check(w, l_min=length)
    assert out["holds"]


def test_length_buckets_are_terciles():
    lengths = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14])
    buckets = obj.length_buckets(lengths)
    assert set(buckets) == {0, 1, 2}
    assert np.all(buckets[:3] == 0)
    assert np.all(buckets[-3:] == 2)
    # Monotone in length.
    assert np.all(np.diff(buckets) >= 0)


def test_saturation_rule_and_endpoint_convention():
    adv = np.array([1.0, -1.0, 1.0, -1.0])
    w = np.array([1.3, 0.7, 1.2, 0.8])
    sat = obj._saturated(adv, w, 0.8, 1.2)
    assert list(sat) == [True, True, False, False]
    # Exact equality is NOT saturated (Prop. 3.4 convention).
    sat = obj._saturated(adv, w, 0.7, 1.3)
    assert not sat.any()


def test_sequence_coeffs_zero_when_saturated():
    w = np.array([1.5, 0.5, 1.0])
    adv = np.array([2.0, -2.0, 1.0])
    per = obj.sequence_coeffs(w, adv, 0.8, 1.2, [4, 4, 4])
    # First two saturated -> 0; third: A*W/(L*N) = 1*1/(4*3).
    assert per[0] == 0.0 and per[1] == 0.0
    assert abs(per[2] - 1.0 / 12.0) < 1e-12


def test_miscoverage_counts_strict_outside():
    w = np.array([0.79, 0.8, 1.0, 1.2, 1.21])
    assert obj.miscoverage(w, 0.8, 1.2) == 0.4  # 0.79 and 1.21 only
