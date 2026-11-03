"""Held-out evaluation: greedy decoding, exact-match pass@1."""

from __future__ import annotations

import numpy as np

from .tasks import partial_credit, reward_of


def evaluate(model, problems, max_new_tokens=24):
    """Greedy-decode every problem; return pass@1 overall + per tier.

    Also reports mean generated length and mean diagnostic partial
    credit (intermediate correctness), which is never trained on.
    """
    prompts = [p.prompt_tokens for p in problems]
    responses, lengths = model.generate(
        prompts, rng=None, max_new_tokens=max_new_tokens, greedy=True)
    correct = []
    partial = []
    per_tier = {}
    for resp, ln, prob in zip(responses, lengths, problems):
        r = reward_of(list(resp[:ln]), prob)
        correct.append(r)
        partial.append(partial_credit(list(resp[:ln]), prob))
        per_tier.setdefault(prob.tier, []).append(r)
    return {
        "n": len(problems),
        "pass_at_1": float(np.mean(correct)),
        "pass_at_1_by_tier": {t: float(np.mean(v))
                              for t, v in per_tier.items()},
        "mean_length": float(np.mean(lengths)),
        "mean_partial_credit": float(np.mean(partial)),
    }
