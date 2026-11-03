"""Tests for the Transformer: backprop vs finite differences,
generation, and determinism."""

import numpy as np

from cspolab import objectives as obj
from cspolab.policy import Transformer, clip_grads
from cspolab.tasks import EOS, PAD, VOCAB_SIZE, make_problems
from cspolab.train import _d_logits_from_coeffs, _response_logp


def _tiny_model(seed=0):
    return Transformer(VOCAB_SIZE, d_model=16, n_heads=2, d_mlp=32,
                       max_len=64, seed=seed)


def _small_batch():
    problems = make_problems(2, seed=11)
    seqs = [p.prompt_tokens + p.response_tokens for p in problems]
    max_len = max(len(s) for s in seqs)
    arr = np.full((2, max_len), PAD, dtype=int)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    rs = np.array([len(p.prompt_tokens) for p in problems])
    rl = np.array([len(p.response_tokens) for p in problems])
    return arr, rs, rl


def _sequence_objective_value(model, sequences, rs, rl, logp_old,
                              advantages, a, b):
    logp, _ = _response_logp(model, sequences, rs, rl)
    _, w = obj.sequence_scores(logp, logp_old, rl)
    return obj.sequence_objective(w, advantages, a, b)


def test_backprop_matches_finite_differences_sequence_objective():
    model = _tiny_model()
    sequences, rs, rl = _small_batch()
    logp_old, _ = _response_logp(model, sequences, rs, rl)
    advantages = np.array([0.7, -0.7])
    a, b = 0.85, 1.15

    def value():
        return _sequence_objective_value(
            model, sequences, rs, rl, logp_old, advantages, a, b)

    # Analytic gradient of J (we ascend J; backward() takes d loss /
    # d logits, so negate the coefficients).
    logp, probs = _response_logp(model, sequences, rs, rl)
    _, w = obj.sequence_scores(logp, logp_old, rl)
    per = obj.sequence_coeffs(w, advantages, a, b, rl)
    coeffs = np.zeros_like(logp)
    for i in range(len(rl)):
        idx = np.arange(rs[i], rs[i] + rl[i])
        coeffs[i, idx] = per[i]
    d_logits = _d_logits_from_coeffs(probs, sequences, rs, rl, coeffs)
    grads = model.backward(d_logits)

    # Finite differences of -J (the loss) on a random subset of entries.
    rng = np.random.default_rng(0)
    eps = 1e-6
    checked = 0
    max_err = 0.0
    for key in model.params:
        arr = model.params[key]
        flat = arr.ravel()
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for ix in idxs:
            old = flat[ix]
            flat[ix] = old + eps
            plus = -value()
            flat[ix] = old - eps
            minus = -value()
            flat[ix] = old
            num = (plus - minus) / (2 * eps)
            max_err = max(max_err, abs(num - grads[key].ravel()[ix]))
            checked += 1
    assert checked >= 30
    assert max_err < 1e-5, f"max finite-difference error {max_err}"


def test_backprop_matches_finite_differences_grpo_objective():
    model = _tiny_model()
    sequences, rs, rl = _small_batch()
    logp_old, _ = _response_logp(model, sequences, rs, rl)
    advantages = np.array([1.2, -0.8])
    eps = 0.2
    mask = np.zeros_like(logp_old, dtype=bool)
    for i in range(len(rl)):
        mask[i, rs[i]:rs[i] + rl[i]] = True

    def value():
        logp, _ = _response_logp(model, sequences, rs, rl)
        ratios = np.exp(logp - logp_old)
        return obj.grpo_objective(ratios, advantages, eps, rl, mask)

    logp, probs = _response_logp(model, sequences, rs, rl)
    ratios = np.exp(logp - logp_old)
    coeffs = obj.grpo_coeffs(ratios, advantages, eps, rl, mask)
    d_logits = _d_logits_from_coeffs(probs, sequences, rs, rl, coeffs)
    grads = model.backward(d_logits)

    rng = np.random.default_rng(1)
    eps_fd = 1e-6
    max_err = 0.0
    checked = 0
    for key in ("W_out", "b_out", "W_1", "tok_emb"):
        arr = model.params[key]
        flat = arr.ravel()
        idxs = rng.choice(flat.size, size=8, replace=False)
        for ix in idxs:
            old = flat[ix]
            flat[ix] = old + eps_fd
            plus = -value()
            flat[ix] = old - eps_fd
            minus = -value()
            flat[ix] = old
            num = (plus - minus) / (2 * eps_fd)
            max_err = max(max_err, abs(num - grads[key].ravel()[ix]))
            checked += 1
    assert checked == 32
    assert max_err < 1e-5, f"max finite-difference error {max_err}"


def test_clipped_gradient_is_zero_when_saturated():
    # Build a case where the response is saturated (A > 0, W > b):
    # the objective gradient must be exactly zero.
    model = _tiny_model()
    sequences, rs, rl = _small_batch()
    logp_old, _ = _response_logp(model, sequences, rs, rl)
    # Fake a behaviour policy far from the current one by shifting
    # logp_old down at response positions only, so W is large.
    mask = np.zeros_like(logp_old, dtype=bool)
    for i in range(len(rl)):
        mask[i, rs[i]:rs[i] + rl[i]] = True
    logp_old = np.where(mask, logp_old - 1.0, logp_old)
    advantages = np.array([1.0, 1.0])
    a, b = 0.8, 1.2
    logp, probs = _response_logp(model, sequences, rs, rl)
    _, w = obj.sequence_scores(logp, logp_old, rl)
    assert np.all(w > b)  # saturated
    per = obj.sequence_coeffs(w, advantages, a, b, rl)
    assert np.all(per == 0.0)


def test_generate_is_deterministic_for_same_seed():
    model = _tiny_model()
    problems = make_problems(4, seed=21)
    prompts = [p.prompt_tokens for p in problems]
    r1 = model.generate(prompts, np.random.default_rng(5), temperature=1.0)
    r2 = model.generate(prompts, np.random.default_rng(5), temperature=1.0)
    assert np.array_equal(r1[0], r2[0])
    assert np.array_equal(r1[1], r2[1])


def test_greedy_generate_stops_at_eos():
    model = _tiny_model()
    problems = make_problems(3, seed=22)
    prompts = [p.prompt_tokens for p in problems]
    responses, lengths = model.generate(prompts, greedy=True)
    for resp, ln in zip(responses, lengths):
        assert EOS not in resp[:ln - 1]  # EOS only as the last token
        if ln < responses.shape[1]:
            assert resp[ln - 1] == EOS
            assert np.all(resp[ln:] == PAD)


def test_forward_logits_finite_with_padding():
    model = _tiny_model()
    arr = np.array([[1, 2, 10, 3, 12, PAD, PAD],
                    [4, 5, 10, 6, 7, 11, 12]])
    logits = model.forward(arr)
    assert logits.shape == (2, 7, VOCAB_SIZE)
    assert np.all(np.isfinite(logits))


def test_clip_grads_caps_global_norm():
    grads = {"a": np.ones((3, 3)) * 10.0, "b": np.ones(4) * 10.0}
    norm = clip_grads(grads, 1.0)
    assert norm > 1.0
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    assert abs(total - 1.0) < 1e-9


def test_log_softmax_sums_to_one():
    model = _tiny_model()
    logits = np.random.default_rng(0).normal(size=(2, 5, VOCAB_SIZE))
    lp = model.log_softmax(logits)
    assert np.allclose(np.exp(lp).sum(axis=-1), 1.0)
