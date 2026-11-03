"""Tests for the arithmetic task generator and reward."""

import numpy as np

from cspolab.tasks import (EOS, PAD, make_problem, make_problems,
                           make_test_set, parse_response, partial_credit,
                           reward_of)


def test_easy_problem_format():
    rng = np.random.default_rng(0)
    p = make_problem(rng, "easy")
    # prompt: a op b op c =
    assert p.prompt_tokens[-1] == 12  # '='
    # response: r1 op c = ans EOS
    assert p.response_tokens[-1] == EOS
    assert p.intermediates[0] + 0 == p.intermediates[0]
    # final answer equals the last intermediate
    assert p.answer == p.intermediates[-1]
    # response text matches the spec example shape
    assert p.response_text.endswith("<EOS>")


def test_intermediate_and_final_answers_correct():
    rng = np.random.default_rng(1)
    for tier in ("easy", "medium", "hard"):
        for _ in range(50):
            p = make_problem(rng, tier)
            # Recompute the chain by hand.
            value = p.operands[0]
            for i, op in enumerate(p.ops):
                value = value + p.operands[i + 1] if op == 0 else value - p.operands[i + 1]
                assert value == p.intermediates[i]
            assert value == p.answer
            # The gold response parses back to the true answer.
            ans, inter, ok = parse_response(p.response_tokens)
            assert ok and ans == p.answer
            assert inter == p.intermediates[:-1]


def test_reward_is_binary_and_exact():
    rng = np.random.default_rng(2)
    p = make_problem(rng, "easy")
    assert reward_of(p.response_tokens, p) == 1
    wrong = list(p.response_tokens)
    # Flip the last answer digit.
    wrong[-2] = (wrong[-2] + 1) % 10
    assert reward_of(wrong, p) == 0
    # No '=' at all -> 0.
    assert reward_of([1, 2, 3, EOS], p) == 0
    # Empty response -> 0.
    assert reward_of([EOS], p) == 0


def test_partial_credit_is_diagnostic():
    rng = np.random.default_rng(3)
    p = make_problem(rng, "hard")
    assert partial_credit(p.response_tokens, p) == 1.0
    # Corrupt the first intermediate digit only: partial credit drops,
    # but the final answer digits are untouched, so the reward stays 1.
    bad = list(p.response_tokens)
    bad[0] = (bad[0] + 1) % 10
    assert partial_credit(bad, p) < 1.0
    assert reward_of(bad, p) == 1
    # Corrupt the final answer digit: reward drops to 0.
    bad2 = list(p.response_tokens)
    bad2[-2] = (bad2[-2] + 1) % 10
    assert reward_of(bad2, p) == 0


def test_length_varies_across_tiers():
    rng = np.random.default_rng(4)
    lens = {tier: [len(make_problem(rng, tier).response_tokens)
                   for _ in range(100)] for tier in ("easy", "medium", "hard")}
    assert min(lens["easy"]) < max(lens["hard"])
    assert max(lens["easy"]) < max(lens["hard"])
    # Every tier has more than one distinct length.
    for tier in lens:
        assert len(set(lens[tier])) > 1
    # Overall range is roughly the documented 6-14+ tokens.
    all_lens = [l for v in lens.values() for l in v]
    assert min(all_lens) >= 6
    assert max(all_lens) <= 24


def test_all_partial_sums_in_range():
    rng = np.random.default_rng(5)
    for _ in range(500):
        p = make_problem(rng, rng.choice(["easy", "medium", "hard"]))
        assert all(0 <= v <= 9999 for v in p.intermediates)


def test_generation_is_deterministic():
    a = make_problems(50, seed=7)
    b = make_problems(50, seed=7)
    assert [p.response_tokens for p in a] == [p.response_tokens for p in b]
    c = make_problems(50, seed=8)
    assert [p.response_tokens for p in a] != [p.response_tokens for p in c]


def test_test_set_is_stratified_and_fixed():
    ts = make_test_set(200, seed=12345)
    assert len(ts) == 200
    from collections import Counter
    counts = Counter(p.tier for p in ts)
    assert counts["easy"] == 80 and counts["medium"] == 70
    assert counts["hard"] == 50
    ts2 = make_test_set(200, seed=12345)
    assert [p.response_tokens for p in ts] == [p.response_tokens for p in ts2]


def test_parse_response_edge_cases():
    # Trailing junk after EOS is ignored; the segment before '=' is
    # "<r1> <op> <operand>" with digits least-significant first, so
    # the intermediate is its leading digit run, reversed.
    ans, inter, ok = parse_response([8, 5, 11, 1, 9, 12, 6, 6, EOS, 1, 2])
    assert ok and ans == 66 and inter == [58]
    # Digits with no '=' -> not ok.
    _, _, ok = parse_response([1, 2, 3, EOS])
    assert not ok
    # '=' with no digits after -> not ok.
    _, _, ok = parse_response([8, 5, 12, EOS])
    assert not ok


def test_prompt_text_roundtrip():
    rng = np.random.default_rng(6)
    p = make_problem(rng, "easy")
    # prompt text is digits/ops/= only, no PAD/EOS.
    assert set(p.prompt_text) <= set("0123456789+-=")
    assert p.prompt_text.endswith("=")
