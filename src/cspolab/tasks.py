"""Variable-length multi-step arithmetic with a verifiable reward.

A problem is a chain of additions/subtractions, e.g. ``47 + 38 - 19 =``.
The policy must generate the intermediate result(s) and then the final
answer as tokens, then EOS:

    prompt:   4 7 + 3 8 - 1 9 =
    response: 5 8 - 9 1 = 6 6 <EOS>

Numbers in the *response* are written least-significant digit first
(``85`` as ``5 8``, ``19`` as ``9 1``); the prompt stays in normal
reading order. This is deliberate, not cosmetic: emitting the
most-significant digit first requires the whole carry chain before
the first output token, and a 1-layer transformer does not learn
that in a night (pilot: after 3000 SFT steps / 384k examples,
teacher-forced accuracy on computed digits was 0.340 and the sampled
reward rate was 0.004 — too sparse for any group-normalised RL
comparison). Least-significant first, each output digit depends only
on the corresponding input digits plus a one-digit carry, which the
model can track autoregressively. The reward, the length variation
and the methods under test are unchanged by the choice.

So the response for a chain with ``n_ops`` operations is, for each
intermediate step ``i = 1 .. n_ops - 1``::

    <r_i digits, least-significant first> <op_{i+1}>
    <operand_{i+1} digits, least-significant first> =

followed by ``<answer digits, least-significant first> <EOS>``. The
reward is 1 iff the final answer digits (the digits after the last
``=``, read least-significant first) are exactly correct, else 0.
Intermediate correctness is recorded as a diagnostic only and is
never used as a training reward.

Vocabulary (15 tokens): digits 0-9, '+', '-', '=', EOS, PAD.

Difficulty tiers, by operand width and chain length:
  easy   -- 2 ops, 2-digit operands
  medium -- 2 ops with 3-digit operands, or 3 ops with 2-digit operands
  hard   -- 3 ops, 3-digit operands

All partial sums are kept in [0, 9999] by rejection sampling, so no
negative or over-long intermediate ever appears. Problem generation is
fully seeded and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- vocabulary -------------------------------------------------------
DIGITS = [str(d) for d in range(10)]
PLUS = 10
MINUS = 11
EQUALS = 12
EOS = 13
PAD = 14
VOCAB_SIZE = 15

TOKEN_TO_CHAR = {d: str(d) for d in range(10)}
TOKEN_TO_CHAR.update({PLUS: "+", MINUS: "-", EQUALS: "=", EOS: "<EOS>",
                      PAD: "<PAD>"})
CHAR_TO_TOKEN = {c: t for t, c in TOKEN_TO_CHAR.items()
                 if c not in ("<EOS>", "<PAD>")}


def digits_of(value):
    """Token ids for the decimal digits of a non-negative integer."""
    return [int(ch) for ch in str(value)]


def digits_of_reversed(value):
    """Token ids for the digits, least-significant first (response order)."""
    return [int(ch) for ch in reversed(str(value))]


def tokens_to_string(tokens):
    return "".join(TOKEN_TO_CHAR[int(t)] for t in tokens)


@dataclass
class Problem:
    prompt_tokens: list
    response_tokens: list
    answer: int
    intermediates: list
    operands: list
    ops: list
    tier: str
    prompt_text: str = ""
    response_text: str = ""

    def __post_init__(self):
        self.prompt_text = tokens_to_string(self.prompt_tokens)
        self.response_text = tokens_to_string(self.response_tokens)


def _sample_chain(rng, n_ops, n_digits):
    """Sample operands/ops so every partial sum stays in [0, 9999]."""
    lo = 10 ** (n_digits - 1)
    hi = 10 ** n_digits - 1
    for _ in range(200):
        operands = [int(rng.integers(lo, hi + 1)) for _ in range(n_ops + 1)]
        ops = [int(rng.integers(0, 2)) for _ in range(n_ops)]  # 0: +, 1: -
        partials = []
        value = operands[0]
        ok = True
        for i, op in enumerate(ops):
            value = value + operands[i + 1] if op == 0 else value - operands[i + 1]
            if value < 0 or value > 9999:
                ok = False
                break
            partials.append(value)
        if ok:
            return operands, ops, partials
    raise RuntimeError("could not sample a valid chain")


def make_problem(rng, tier):
    """Draw one problem of the given tier from ``rng``."""
    if tier == "easy":
        n_ops, n_digits = 2, 2
    elif tier == "medium":
        n_ops, n_digits = (3, 2) if rng.random() < 0.5 else (2, 3)
    elif tier == "hard":
        n_ops, n_digits = 3, 3
    else:
        raise ValueError(f"unknown tier {tier!r}")

    operands, ops, partials = _sample_chain(rng, n_ops, n_digits)
    op_tokens = [PLUS if op == 0 else MINUS for op in ops]

    prompt = []
    for i, operand in enumerate(operands):
        prompt.extend(digits_of(operand))
        if i < n_ops:
            prompt.append(op_tokens[i])
    prompt.append(EQUALS)

    response = []
    for i in range(n_ops - 1):
        response.extend(digits_of_reversed(partials[i]))
        response.append(op_tokens[i + 1])
        response.extend(digits_of_reversed(operands[i + 2]))
        response.append(EQUALS)
    response.extend(digits_of_reversed(partials[-1]))
    response.append(EOS)

    return Problem(prompt_tokens=prompt, response_tokens=response,
                   answer=partials[-1], intermediates=partials,
                   operands=operands, ops=ops, tier=tier)


def make_problems(n, seed, tier_weights=None):
    """Draw ``n`` problems; tier mix given by ``tier_weights``.

    Default mix: easy 0.5, medium 0.35, hard 0.15. The generator is a
    single seeded stream, so the same ``(n, seed)`` always yields the
    same problems in the same order.
    """
    if tier_weights is None:
        tier_weights = {"easy": 0.5, "medium": 0.35, "hard": 0.15}
    tiers = list(tier_weights)
    probs = np.array([tier_weights[t] for t in tiers], dtype=np.float64)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(tiers, size=n, p=probs)
    return [make_problem(rng, tier) for tier in chosen]


def make_test_set(n=200, seed=12345):
    """Held-out test set, stratified by tier (easy/medium/hard)."""
    rng = np.random.default_rng(seed)
    counts = {"easy": int(round(0.4 * n)), "medium": int(round(0.35 * n)),
              "hard": n - int(round(0.4 * n)) - int(round(0.35 * n))}
    problems = []
    for tier in ("easy", "medium", "hard"):
        problems.extend(make_problem(rng, tier) for _ in range(counts[tier]))
    return problems


def parse_response(tokens):
    """Parse generated response tokens.

    Returns ``(answer, intermediates, ok)`` where ``answer`` is the
    integer after the last ``=`` (or None if unparseable),
    ``intermediates`` is the list of integers after each earlier ``=``
    (may be shorter than the true chain if the model cut corners), and
    ``ok`` says whether the format was well-formed at all (at least one
    ``=`` followed by at least one digit before EOS / end).
    """
    toks = [int(t) for t in tokens]
    # Drop everything from EOS onward; PADs never appear in a response.
    if EOS in toks:
        toks = toks[:toks.index(EOS)]
    eq_positions = [i for i, t in enumerate(toks) if t == EQUALS]
    if not eq_positions:
        return None, [], False
    segments = []
    prev = -1
    for pos in eq_positions:
        segments.append(toks[prev + 1:pos])
        prev = pos
    tail = toks[eq_positions[-1] + 1:]
    if not tail or any(t < 0 or t > 9 for t in tail):
        return None, [], False
    # Response digits are least-significant first; reverse to read.
    answer = int("".join(str(t) for t in reversed(tail))) if tail else None
    intermediates = []
    for seg in segments:
        # A segment is "<r_i> <op> <operand>" (digits reversed): the
        # intermediate value is the leading run of digit tokens.
        lead = []
        for t in seg:
            if 0 <= t <= 9:
                lead.append(t)
            else:
                break
        intermediates.append(int("".join(str(t) for t in reversed(lead)))
                             if lead else None)
    return answer, intermediates, True


def reward_of(tokens, problem):
    """Binary verifiable reward: 1 iff the final answer is exact."""
    answer, _, ok = parse_response(tokens)
    if not ok or answer is None:
        return 0
    return 1 if answer == problem.answer else 0


def partial_credit(tokens, problem):
    """Diagnostic only: fraction of true intermediates reproduced.

    Compares the parsed intermediate values against the true chain,
    position by position. Never used as a training reward.
    """
    _, intermediates, ok = parse_response(tokens)
    if not ok or not problem.intermediates[:-1]:
        return 0.0
    true_inter = problem.intermediates[:-1]
    hits = sum(1 for got, want in zip(intermediates, true_inter)
               if got == want)
    return hits / len(true_inter)
