"""Training: SFT warm-start, rollouts, and GRPO/GSPO/FSPO/CSPO updates.

Why an SFT warm-start: a randomly initialised policy scores ~0 on
exact-match arithmetic, so every group has zero reward variance and
the group-normalised advantages are all zero -- there is no RL signal
at all. The paper's base model (Qwen3-1.7B) already does arithmetic;
the honest small-model analogue is to warm-start with supervised
fine-tuning on the same task, then run every RL method from the same
shared SFT checkpoint. The SFT phase is cross-entropy on response
tokens only, with fresh seeded problems each step. It is easy-only
(see docs/RESEARCH.md): mixed-tier SFT never learns the arithmetic
algorithm, and a mixed second phase does not transfer to medium/hard,
so the warm-start is the easy-only checkpoint and RL trains on the
full tier mix from there.

One RL step, for a batch of ``B`` prompts and ``G`` samples each
(``N = B * G`` responses):
  1. sample ``G`` responses per prompt at temperature 1.0;
  2. compute the behaviour log-probs, binary rewards, and
     group-normalised advantages;
  3. run ``E`` optimiser updates. For CSPO the band ``(a_t, b_t)`` is
     calibrated once at the start of the step from the rolling buffer
     and held fixed through all ``E`` updates; the scores computed at
     each update are retained and appended to the buffer as one block
     when the step completes. For cspo-len each length tercile has
     its own buffer and band.
"""

from __future__ import annotations

import time

import numpy as np

from . import objectives as obj
from .evaluate import evaluate
from .policy import Adam, Transformer, clip_grads
from .tasks import EOS, PAD, VOCAB_SIZE, make_problems, partial_credit, reward_of


def _pad_sequences(seqs, pad_value=PAD):
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), pad_value, dtype=int)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def _response_mask(response_start, response_len, n_cols):
    """Bool ``(N, T)`` mask marking response token positions."""
    n = len(response_start)
    mask = np.zeros((n, n_cols), dtype=bool)
    for i in range(n):
        mask[i, response_start[i]:response_start[i] + response_len[i]] = True
    return mask


def _response_logp(model, sequences, response_start, response_len):
    """Log-probs of the tokens at response positions.

    ``sequences`` is ``(N, T)`` (prompt + response, PAD-padded);
    returns ``logp`` of shape ``(N, T)`` with zeros at non-response
    positions, plus the per-position softmax probabilities (for
    entropy and for the ``d log p / d logits`` Jacobian).
    """
    logits = model.forward(sequences)
    logp_all = model.log_softmax(logits)
    n, t = sequences.shape
    logp = np.zeros((n, t))
    probs = np.exp(logp_all)
    for i in range(n):
        s, ln = response_start[i], response_len[i]
        idx = np.arange(s, s + ln)
        logp[i, idx] = logp_all[i, idx - 1, sequences[i, idx]]
    return logp, probs


def _d_logits_from_coeffs(probs, sequences, response_start, response_len,
                          coeffs):
    """Gradient of an objective w.r.t. logits, from per-token coeffs.

    ``coeffs`` is ``(N, T)`` with ``dJ / d log p`` at response
    positions (0 elsewhere). The loss is ``-J``, so
    ``d(-J)/d logits = -coeff * (one_hot - p)`` at the position that
    *predicts* each response token (one position earlier).
    """
    n, t, v = probs.shape
    d_logits = np.zeros((n, t, v))
    for i in range(n):
        s, ln = response_start[i], response_len[i]
        idx = np.arange(s, s + ln)
        pred = idx - 1
        c = coeffs[i, idx]
        one_hot = np.zeros((ln, v))
        one_hot[np.arange(ln), sequences[i, idx]] = 1.0
        d_logits[i, pred, :] = -c[:, None] * (one_hot - probs[i, pred, :])
    return d_logits


def sft_pretrain(model, seed, steps, batch_size=64, lr=1e-3,
                 log_every=100, tier_weights=None):
    """Supervised warm-start on fresh seeded problems.

    Returns a list of ``(step, loss, greedy_acc)`` probe rows; the
    greedy accuracy is measured on 64 fresh problems every
    ``log_every`` steps (and at step 0). ``tier_weights`` selects the
    tier mix (default: the standard easy/medium/hard mix); pass
    ``{"easy": 1.0}`` for an easy-only curriculum phase.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    probe = make_problems(64, seed=999, tier_weights=tier_weights)
    rows = []
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            acc = evaluate(model, probe)["pass_at_1"]
            rows.append({"step": step, "greedy_acc_64": acc})
        if step == steps:
            break
        problems = make_problems(batch_size, seed=int(rng.integers(2 ** 31)),
                                 tier_weights=tier_weights)
        seqs = [p.prompt_tokens + p.response_tokens for p in problems]
        sequences = _pad_sequences(seqs)
        resp_start = np.array([len(p.prompt_tokens) for p in problems])
        resp_len = np.array([len(p.response_tokens) for p in problems])
        logits = model.forward(sequences)
        logp_all = model.log_softmax(logits)
        n, t, v = logits.shape
        # Cross-entropy on response tokens only.
        d_logits = np.zeros((n, t, v))
        total_loss = 0.0
        total_tokens = 0
        for i in range(n):
            s, ln = resp_start[i], resp_len[i]
            idx = np.arange(s, s + ln)
            pred = idx - 1
            targets = sequences[i, idx]
            lp = logp_all[i, pred, targets]
            total_loss -= float(lp.sum())
            total_tokens += ln
            one_hot = np.zeros((ln, v))
            one_hot[np.arange(ln), targets] = 1.0
            d_logits[i, pred, :] = (np.exp(logp_all[i, pred, :]) - one_hot) / total_tokens
        grads = model.backward(d_logits)
        clip_grads(grads, 1.0)
        opt.update(grads)
        if step % log_every == 0:
            rows[-1]["loss"] = total_loss / total_tokens
    return rows


def rollout(model, problems, group_size, rng, max_new_tokens=24):
    """Sample ``group_size`` responses per prompt at temperature 1.0.

    Returns a dict with the padded full sequences, response starts /
    lengths, behaviour log-probs at response positions, binary rewards
    and diagnostic partial credit.
    """
    prompts = [p.prompt_tokens for p in problems for _ in range(group_size)]
    responses, lengths = model.generate(
        prompts, rng, max_new_tokens=max_new_tokens, temperature=1.0)
    sequences = _pad_sequences(
        [list(pr) + list(resp[:ln]) for pr, resp, ln in
         zip(prompts, responses, lengths)])
    resp_start = np.array([len(pr) for pr in prompts])
    resp_len = lengths.copy()
    logp_old, probs_old = _response_logp(model, sequences, resp_start,
                                         resp_len)
    rewards = np.array([
        reward_of(list(resp[:ln]), prob)
        for resp, ln, prob in zip(responses, lengths,
                                  [problems[i // group_size]
                                   for i in range(len(prompts))])
    ], dtype=np.float64)
    partial = np.array([
        partial_credit(list(resp[:ln]), prob)
        for resp, ln, prob in zip(responses, lengths,
                                  [problems[i // group_size]
                                   for i in range(len(prompts))])
    ], dtype=np.float64)
    return {
        "sequences": sequences, "response_start": resp_start,
        "response_len": resp_len, "logp_old": logp_old,
        "probs_old": probs_old,
        "rewards": rewards, "partial_credit": partial,
    }


def _entropy_at_response(probs, response_start, response_len):
    """Mean per-token entropy of the policy at response positions."""
    n, t, _ = probs.shape
    ents = []
    for i in range(n):
        s, ln = response_start[i], response_len[i]
        pred = np.arange(s, s + ln) - 1
        p = probs[i, pred, :]
        ent = -(p * np.log(p + 1e-12)).sum(axis=1)
        ents.append(ent)
    return float(np.mean(np.concatenate(ents)))


def run_training(method, seed, steps, init_params, test_set, config):
    """Run one RL training run from a shared SFT checkpoint.

    ``method`` is one of ``grpo``, ``gspo``, ``fspo``, ``cspo``,
    ``unclipped``, ``cspo-len``. ``config`` carries ``B`` (prompts
    per step), ``G`` (samples per prompt), ``E`` (updates per step),
    ``lr``, ``alpha`` (CSPO), ``K`` (CSPO buffer blocks),
    ``warm_c`` (CSPO warm-start half-width) and ``eps`` (GRPO clip).
    Returns ``(result, model)`` where ``result`` holds the per-step
    diagnostics, mid-training and final evals, and the final band
    state.
    """
    b_prompts = config["B"]
    group_size = config["G"]
    n_updates = config["E"]
    alpha = config.get("alpha", 0.1)
    k_blocks = config.get("K", 2)
    warm_c = config.get("warm_c", 0.2)
    eps = config.get("eps", 0.2)

    model = Transformer(VOCAB_SIZE, seed=seed)
    model.params = {k: v.copy() for k, v in init_params.items()}
    opt = Adam(model.params, lr=config["lr"])
    rng = np.random.default_rng(seed)

    buffers = {"all": obj.RollingBuffer(k=k_blocks)}
    if method == "cspo-len":
        buffers = {b: obj.RollingBuffer(k=k_blocks) for b in range(3)}

    diagnostics = []
    evals = {}
    t_start = time.time()
    for step in range(steps):
        problems = make_problems(b_prompts, seed=int(rng.integers(2 ** 31)))
        batch = rollout(model, problems, group_size, rng)
        seqs = batch["sequences"]
        rs, rl = batch["response_start"], batch["response_len"]
        logp_old = batch["logp_old"]
        rewards = batch["rewards"]
        advantages = obj.group_advantages(rewards, group_size)
        buckets = obj.length_buckets(rl) if method == "cspo-len" else None

        # Calibrate the band once for this step; hold it fixed.
        if method == "cspo":
            band_a, band_b = obj.calibrate_band(
                buffers["all"].values(), alpha=alpha, warm_c=warm_c)
        elif method == "cspo-len":
            band_a = np.zeros(len(rl))
            band_b = np.zeros(len(rl))
            for bkt in range(3):
                a_b, b_b = obj.calibrate_band(
                    buffers[bkt].values(), alpha=alpha, warm_c=warm_c)
                band_a[buckets == bkt] = a_b
                band_b[buckets == bkt] = b_b
        elif method == "gspo":
            band_a, band_b = 0.8, 1.2
        elif method == "fspo":
            band_a, band_b = obj.fspo_band(rl)
        elif method == "unclipped":
            band_a, band_b = 0.0, np.inf
        elif method == "grpo":
            band_a, band_b = None, None
        else:
            raise ValueError(f"unknown method {method!r}")

        step_scores = []
        step_diag = None
        resp_mask = _response_mask(rs, rl, seqs.shape[1])
        for k in range(n_updates):
            if k == 0:
                # The policy still equals the behaviour policy, so the
                # log-probs are exactly the rollout ones -- no forward.
                logp, probs = batch["logp_old"], batch["probs_old"]
            else:
                logp, probs = _response_logp(model, seqs, rs, rl)
            if method == "grpo":
                ratios = np.exp(logp - logp_old)
                j_val = obj.grpo_objective(ratios, advantages, eps, rl,
                                           resp_mask)
                coeffs = obj.grpo_coeffs(ratios, advantages, eps, rl,
                                         resp_mask)
                w = None
            else:
                _, w = obj.sequence_scores(logp, logp_old, rl)
                j_val = obj.sequence_objective(w, advantages, band_a, band_b)
                per_resp = obj.sequence_coeffs(w, advantages, band_a, band_b, rl)
                coeffs = np.zeros_like(logp)
                for i in range(len(rl)):
                    idx = np.arange(rs[i], rs[i] + rl[i])
                    coeffs[i, idx] = per_resp[i]
                step_scores.append(w.copy())
            d_logits = _d_logits_from_coeffs(probs, seqs, rs, rl, coeffs)
            grads = model.backward(d_logits)
            grad_norm = clip_grads(grads, 1.0)
            opt.update(grads)

            if method == "grpo":
                sat = float(np.mean(obj._saturated(
                    advantages[:, None], ratios,
                    1.0 - eps, 1.0 + eps)[resp_mask]))
                step_diag = {
                    "step": step, "reward": float(rewards.mean()),
                    "partial_credit": float(batch["partial_credit"].mean()),
                    "entropy": _entropy_at_response(probs, rs, rl),
                    "mean_length": float(rl.mean()),
                    "band_a": 1.0 - eps, "band_b": 1.0 + eps,
                    "band_width": 2.0 * eps,
                    "sat_frac": sat, "miscoverage": None,
                    "miscoverage_by_bucket": None,
                    "j_value": j_val, "grad_norm": grad_norm,
                    "prop33_gap": None, "prop33_bound": None,
                    "prop33_holds": None, "lemma32_e_w2": None,
                    "lemma32_bound": None, "lemma32_holds": None,
                }
            else:
                l_min = int(rl.min())
                p33 = obj.prop33_check(w, advantages, band_a, band_b,
                                       l_min, group_size)
                l32 = obj.lemma32_check(w, l_min)
                if k >= 1:
                    misc = obj.miscoverage(w, band_a, band_b)
                    if method == "cspo-len":
                        misc_bkt = [obj.miscoverage(w[buckets == bkt],
                                                    band_a[buckets == bkt],
                                                    band_b[buckets == bkt])
                                    if np.any(buckets == bkt) else None
                                    for bkt in range(3)]
                    else:
                        misc_bkt = None
                else:
                    misc, misc_bkt = None, None
                step_diag = {
                    "step": step, "reward": float(rewards.mean()),
                    "partial_credit": float(batch["partial_credit"].mean()),
                    "entropy": _entropy_at_response(probs, rs, rl),
                    "mean_length": float(rl.mean()),
                    "band_a": (float(np.mean(band_a)) if np.ndim(band_a)
                               else float(band_a)),
                    "band_b": (float(np.mean(band_b)) if np.ndim(band_b)
                               else float(band_b)),
                    "band_width": float(np.mean(
                        np.asarray(band_b) - np.asarray(band_a))),
                    "sat_frac": obj.saturation_fraction(
                        w, advantages, band_a, band_b),
                    "miscoverage": misc,
                    "miscoverage_by_bucket": misc_bkt,
                    "j_value": j_val, "grad_norm": grad_norm,
                    "prop33_gap": p33["gap"], "prop33_bound": p33["bound"],
                    "prop33_holds": p33["holds"],
                    "lemma32_e_w2": l32["e_w2"],
                    "lemma32_bound": l32["bound"],
                    "lemma32_holds": l32["holds"],
                }
                if method == "cspo-len":
                    step_diag["band_a_by_bucket"] = [
                        float(band_a[buckets == bkt][0])
                        if np.any(buckets == bkt) else None
                        for bkt in range(3)]
                    step_diag["band_b_by_bucket"] = [
                        float(band_b[buckets == bkt][0])
                        if np.any(buckets == bkt) else None
                        for bkt in range(3)]
        diagnostics.append(step_diag)

        # Retain this step's scores as one block (CSPO variants only).
        if method == "cspo":
            buffers["all"].add_block(np.concatenate(step_scores))
        elif method == "cspo-len":
            all_scores = np.concatenate(step_scores)
            for bkt in range(3):
                # Each update's scores repeat the same responses, so
                # the per-bucket block keeps the k-th update's entries
                # in order; take the entries of that bucket from the
                # concatenated list by tiling the bucket mask.
                mask = np.tile(buckets == bkt, len(step_scores))
                buffers[bkt].add_block(all_scores[mask])

        if step == steps // 2:
            evals["mid"] = evaluate(model, test_set)
    evals["final"] = evaluate(model, test_set)

    if method == "cspo":
        final_band = obj.calibrate_band(buffers["all"].values(),
                                        alpha=alpha, warm_c=warm_c)
    elif method == "cspo-len":
        final_band = [obj.calibrate_band(buffers[b].values(), alpha=alpha,
                                         warm_c=warm_c) for b in range(3)]
    else:
        final_band = None

    result = {
        "method": method, "seed": seed, "steps": steps,
        "config": {k: v for k, v in config.items()},
        "diagnostics": diagnostics, "evals": evals,
        "final_band": final_band,
        "wall_seconds": time.time() - t_start,
    }
    return result, model
