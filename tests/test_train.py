"""Smoke tests for the training loop: determinism and a short run."""

import numpy as np

from cspolab.policy import Transformer, copy_params
from cspolab.tasks import VOCAB_SIZE, make_problems, make_test_set
from cspolab.train import run_training, sft_pretrain


def test_sft_is_deterministic_for_same_seed():
    m1 = Transformer(VOCAB_SIZE, seed=0)
    m2 = Transformer(VOCAB_SIZE, seed=0)
    sft_pretrain(m1, seed=3, steps=3, batch_size=8, log_every=100)
    sft_pretrain(m2, seed=3, steps=3, batch_size=8, log_every=100)
    for k in m1.params:
        assert np.array_equal(m1.params[k], m2.params[k])


def test_sft_reduces_loss_on_tiny_run():
    model = Transformer(VOCAB_SIZE, seed=0)
    rows = sft_pretrain(model, seed=4, steps=30, batch_size=16,
                        log_every=10)
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_rollout_logp_matches_fresh_forward():
    """The k=0 update reuses the rollout log-probs; they must be exact."""
    model = Transformer(VOCAB_SIZE, seed=0)
    rng = np.random.default_rng(0)
    problems = make_problems(2, seed=31)
    from cspolab.train import _response_logp, rollout
    batch = rollout(model, problems, 2, rng)
    logp, probs = _response_logp(model, batch["sequences"],
                                 batch["response_start"],
                                 batch["response_len"])
    assert np.array_equal(logp, batch["logp_old"])
    assert np.array_equal(probs, batch["probs_old"])


def test_short_cspo_run_completes_and_records_diagnostics():
    model = Transformer(VOCAB_SIZE, seed=0)
    sft_pretrain(model, seed=5, steps=5, batch_size=8, log_every=100)
    init = copy_params(model.params)
    test_set = make_test_set(20, seed=77)
    config = {"B": 2, "G": 4, "E": 2, "lr": 3e-4,
              "alpha": 0.1, "K": 2, "warm_c": 0.2, "eps": 0.2}
    result, _ = run_training("cspo", seed=0, steps=3, init_params=init,
                             test_set=test_set, config=config)
    assert len(result["diagnostics"]) == 3
    d0 = result["diagnostics"][0]
    # First step uses the warm-start band.
    assert d0["band_a"] == 0.8 and d0["band_b"] == 1.2
    # Miscoverage is only defined from the second update onward.
    assert d0["miscoverage"] is not None
    assert 0.0 <= d0["miscoverage"] <= 1.0
    assert result["evals"]["final"]["n"] == 20
    assert 0.0 <= result["evals"]["final"]["pass_at_1"] <= 1.0


def test_short_run_for_each_method():
    model = Transformer(VOCAB_SIZE, seed=0)
    sft_pretrain(model, seed=5, steps=5, batch_size=8, log_every=100)
    init = copy_params(model.params)
    test_set = make_test_set(20, seed=77)
    config = {"B": 2, "G": 4, "E": 2, "lr": 3e-4,
              "alpha": 0.1, "K": 2, "warm_c": 0.2, "eps": 0.2}
    for method in ("grpo", "gspo", "fspo", "unclipped", "cspo-len"):
        result, _ = run_training(method, seed=0, steps=2,
                                 init_params=init, test_set=test_set,
                                 config=config)
        assert len(result["diagnostics"]) == 2
        assert result["evals"]["final"]["n"] == 20
        if method == "cspo-len":
            assert len(result["final_band"]) == 3
