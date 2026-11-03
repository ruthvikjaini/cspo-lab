# Research notes — cspo-lab

## Trending check (done by the parent, 2026-11-03 — recorded here, not redone)

GitHub daily trending 2026-11-03: vercel-labs/agent-desktop,
claude-mem, Scrapling, Hyprland, claude-code-game-studios,
FlashML-org/FreeToken, Kronos, kornia, colibri, nanochat, heretic,
skill packs. Weekly: hallmark, claude-mem, impeccable, claude-seo,
superpowers, thinkrail, security-audit-skill, TradingAgents, ghostty,
ECC, hyperframes. Verdict: no trending repo with a genuine measurable
one-night CPU weakness — nanochat / FreeToken / colibri / Kronos need
GPU training or hardware, skill/prompt packs have no benchmarkable
code, TradingAgents is a full platform. So paper fallback per the
standing rule.

## The paper

Yang, Tian, Ren, Li & Peng, "Conformal Prediction-Enabled
Sequence-level Policy Optimization for Reasoning Language Models"
(arXiv:2610.25092, submitted 18 Oct 2026, Peking University / Beihang
/ Nanjing University). Verified by the parent on the arXiv abstract +
full HTML on 2026-11-03.

### The problem: a unit mismatch in GRPO

GRPO clips token-level ratios `r_{i,l}` while the advantage is a
response-level scalar — the clip acts on a different unit than the
credit. GSPO (Zheng et al. 2025) fixes the unit: sequence log-ratio
`S_i = sum_l log r_{i,l}`, power score `W_i = exp(S_i / L_i)` (the
geometric mean of the token ratios), fixed clip band
`(1 - eps_low, 1 + eps_high)`, objective

    J_seq = E[ (1/G) sum_i min{W_i A_i, clip(W_i, 1-eps_low, 1+eps_high) A_i} ]

with group-normalised advantages `A_i = (R_i - mean R) / (std R + eps_A)`.

### FSPO: length fairness

FSPO (Mao et al. 2025) clips the sequence log-ratio `S` with a band
scaling as `sqrt(L)` — `|S_i| <= 0.2 sqrt(L_i)`, equivalently
`W_i` in `exp(+/- 0.2 / sqrt(L_i))`. Longer responses get a narrower
band in `W`, so the same amount of per-token drift is treated the
same regardless of length.

### CSPO: conformal calibration of the band (Algorithm 1)

Before each training step `t`, use a rolling buffer `B_t` of retained
training scores `W = exp(S/L)` from the most recent `K` completed
blocks (paper main setting `alpha = 0.1`, `K = 2`). Estimate the
lower/upper empirical quantiles `q_lo = Q_{alpha_lo}(B_t)`,
`q_hi = Q_{1-alpha_hi}(B_t)` with `alpha_lo = alpha_hi = alpha/2`,
anchor at `W = 1`: `a_t = min(q_lo, 1)`, `b_t = max(q_hi, 1)`. Hold
`(a_t, b_t)` fixed through the `E_t` optimizer updates of that
rollout batch. If the buffer is empty, warm-start with `(1 - c, 1 + c)`,
`c = 0.2`. Scores computed on the training batch at each update `k`
are retained for subsequent steps (repeat scoring of the same
response counts as separate entries). CSPO reduces to fixed-band
GSPO when the endpoints are fixed.

### Theory verified numerically in this lab

- Theorem B.2 (pathwise bound): miscoverage of fresh scores against
  the calibrated band is at most `alpha + 2 eps_t + 2 D_t` (CDF
  discrepancy + drift). The lab checks the realised miscoverage at
  each update `k >= 1` against the target `alpha`.
- Lemma 3.2: `E[W^2] <= 1 + 2/L_min`.
- Proposition 3.3 (objective distortion):
  `0 <= J_G - J_C <= A_max [ sqrt((1+2/L_min) p+) + a p- ]` where
  `p+ = P(W > b)`, `p- = P(W < a)`, `A_max = sqrt(G)` under the
  population-std advantage convention.
- Proposition 3.4 (gradient error): the gradient rule implemented
  here — zero gradient when saturated (`A > 0, W > b` or `A < 0,
  W < a`), unclipped derivative at exact equality — is the paper's
  endpoint convention. The lab verifies the clipped-objective
  gradient against finite differences.
- The paper explicitly notes calibration is marginal over response
  length — calibration conditional on length requires additional
  assumptions. That open point is the lab's extension: length-
  stratified CSPO (`cspo-len`), with a separate rolling buffer and
  band per length tercile.

### Paper results (the paper's numbers, on Qwen3-1.7B — not this lab's)

Math benchmarks, average Pass@1: CSPO 34.11 vs GSPO 33.36, GRPO
30.36, DAPO 31.42, base 20.83. ALFWorld: CSPO 64.06 vs GSPO 31.25,
GRPO 52.34, DAPO 53.12. Alpha ablation on DAPO-Math-17k: alpha 0.1
-> 34.11, 0.2 -> 33.50, 0.4 -> 32.37. The band contracts early in
training then varies in a narrower range; endpoints follow different
trajectories per tail.

## Rejected candidates

- Liu et al., "No Free Lunch in RLVR" (arXiv:2610.39295) — held from
  Day 9; theorem-heavy, same RLVR cluster as Days 1/7, kept held.
- CoEvo-RSI — too vague for one honest CPU night.
- BookNavigator RL (arXiv:2610.39671) — needs a real LLM + a textbook
  corpus.

## Design decisions

- **Testbed.** Variable-length multi-step arithmetic with a
  verifiable binary reward (1 iff the final answer digits are
  exactly correct). Problems are chains of 2–3 additions/subtractions
  over 2–3 digit operands; the policy must emit the intermediate
  result(s) then the final answer, then EOS, e.g. prompt
  `47 + 38 - 19 =`, response `85 - 19 = 66 <EOS>`. Output lengths run
  roughly 6–21 tokens, which is exactly what sequence-level clipping
  and length fairness are about. Partial credit (intermediate
  correctness) is recorded as a diagnostic only and never trained on.
  Fixed train/test split by seed; held-out test set of 200 problems
  (80 easy / 70 medium / 50 hard).
- **Vocab.** 15 tokens: digits 0–9, `+`, `-`, `=`, EOS, PAD.
- **Policy.** 1-layer, 4-head, width 64, ReLU MLP (hidden 256), no
  LayerNorm, learned positional embeddings — the Day 9 lab's scaffold
  (see `softmax-cliff-lab`, same pattern), but returning logits at
  every position instead of only the final one, because policy
  gradients need every generated token's log-probability. All
  parameters and arithmetic are float64, for byte-identical reruns.
- **Sampling.** Temperature 1.0 for rollouts (documented choice,
  fixed), greedy decoding for evaluation.
- **Optimiser.** Plain Adam, lr 3e-4, betas (0.9, 0.999), eps 1e-8,
  no weight decay; gradients clipped by global norm 1.0.
- **Quantile convention.** Empirical quantiles use the inverse
  empirical CDF: `np.quantile(values, q, method="lower")` (the
  smallest order statistic whose empirical CDF is >= `q`). This is
  documented because the band endpoints depend on it at small buffer
  sizes.
- **SFT warm-start.** A randomly initialised policy scores ~0 on
  exact-match arithmetic, so every group has zero reward variance
  and the group-normalised advantages are all zero — there is no RL
  signal at all. The paper's base model (Qwen3-1.7B) already does
  arithmetic; the honest small-model analogue is a supervised
  warm-start on the same task, with every RL method run from the
  same shared SFT checkpoint. See the pilot notes below.
- **`E = 2` updates per rollout batch**, `B = 16` prompts/step,
  `G = 8` samples/prompt (see pilot timing for the final grid size).

## Pilot notes

### Pilot 1 — SFT does not learn at lr 1e-3, batch 64 (2026-11-03, ~00:20 ET)

SFT 600 steps, batch 64, lr 1e-3: loss 2.71 -> 1.70, but greedy
pass@1 0.0 on the 200-problem test set at every probe (steps 0–600)
and mean partial credit 0.0025. The model learns the format (loss
well below the `ln 15 = 2.71` uniform baseline) but not the
arithmetic: emitting the most-significant digit of a sum first
requires the full carry chain before the first output token, which a
1-layer transformer learns slowly. Consequence: rollout reward was
0.0 at every CSPO pilot step, so no method could show any signal.
The warm-start had to be fixed before any RL comparison was
meaningful.

### Pilot 2 — SFT diagnostic at lr 3e-3, batch 128, 3000 steps (2026-11-03, ~01:00 ET)

SFT 3000 steps (384k examples), lr 3e-3, batch 128: loss 2.70 -> 0.89,
but greedy pass@1 still 0.000 and mean partial credit 0.005. The
diagnostic split explains why: teacher-forced accuracy on computed
digits was 0.340 (451/1326) while structural/copy tokens were 0.984
(1150/1169) — the model learned the format and the copying almost
perfectly, but not the arithmetic. Sampled reward rate at temperature
1.0 was 0.0039 (1 correct in 256 samples). At that rate a rollout
batch of 128 responses would contain ~0.5 correct answers, so almost
every group would have zero reward variance and zero advantages:
no RL method could show a signal. Conclusion: the
most-significant-first format is not viable for a one-night
small-model lab. Fix: emit response numbers least-significant digit
first (see the testbed note in `src/cspolab/tasks.py`). The prompt
stays in normal reading order; only generated numbers are reversed.
The reward, length variation, tiers and methods are unchanged.

### Pilot 3 — SFT on the least-significant-first format (2026-11-03, ~01:15 ET)

SFT 1500 steps (192k examples), lr 3e-3, batch 128: loss 2.72 -> 0.98,
greedy pass@1 0.005, sampled reward rate 0.0000 (0 correct in 256
samples). The format change did **not** fix it. Diagnostic split on
the saved checkpoint (200 problems, teacher-forced): intermediate
digits 0.306, answer digits 0.247, reversed operand copies 0.957,
operators/`=`/EOS 0.985. So the reversed copy works fine — the
bottleneck is purely the arithmetic computation (digit-wise
addition/subtraction with carry/borrow), which the 1-layer model
only gets ~25–31% right. Both digit orders plateau at the same
~30% computed-digit accuracy. The 1-layer width-64 model appears to
lack the compositional depth for the carry chain in one
attention+MLP pass.

### Pilot 4 — easy-only SFT (2026-11-03, ~01:30 ET)

Can the 1-layer model learn 2-digit addition at all when it is the
only thing it has to learn? 1500 steps, batch 128, lr 3e-3, easy
tier only (2-digit, 2 ops): loss 2.73 -> 0.13, greedy accuracy on
the 64-problem probe 0.000 -> 0.719, with a sharp phase transition
between steps 500 and 750 (0.047 -> 0.422). Held-out easy pass@1
**0.838**, partial credit 0.988, sampled reward rate on easy
**0.707**. So the model *can* learn the algorithm — the mixed tier
distribution was polluting learning. Fix: curriculum SFT (phase 1
easy-only, phase 2 mixed). This also explains pilots 1–3: the
3-digit/3-op problems in the mix kept the model from ever finding
the 2-digit algorithm.

### Pilot 5 — curriculum phase 2, mixed tiers (2026-11-03, ~01:45 ET)

Continue from the easy-only checkpoint on the standard tier mix,
1500 steps: loss 24.35 -> 0.72 (the 24.35 is the first mixed batch:
the model is confidently wrong on unseen 3-digit/3-op problems).
Re-evaluated with the fixed `generate`, the phase-2 checkpoint
scores pass@1 0.310 (easy 0.78, medium 0.0, hard 0.0), sampled
reward 0.227 — i.e. the mixed phase did **not** transfer to
medium/hard and slightly diluted easy (0.84 -> 0.78). Conclusion:
the shipped warm-start is the easy-only checkpoint (pass@1 0.335,
easy 0.84, sampled reward 0.281); RL trains on the full tier mix
from there, so medium/hard improvement must come from RL
generalisation, not from SFT.

### Pilot 6 — `generate` padding bug (2026-11-03, ~01:50 ET)

While re-evaluating checkpoints, the full-test-set eval of the
easy-only checkpoint returned pass@1 0.0 with mean length 3.04,
while the easy-subset eval of the same model returned 0.8375.
Root cause: `generate` right-padded short prompts up to the longest
prompt, inserting PAD tokens *between the prompt and the generated
response*. The model never saw that layout in training (responses
start immediately after the prompt), so the first generated token
was predicted from a PAD position and greedy decoding derailed.
This corrupted every mixed-tier greedy/sampled eval in pilots 1–5
(the mixed probes and the phase-2 "0.0" above); easy-only evals
survived because easy prompts are near-uniform in length. Fixed by
tracking per-row current lengths and right-padding only *after*
each row's last real token. After the fix: full eval of the
easy-only checkpoint is pass@1 0.335 (easy 0.84, medium 0.0,
hard 0.0), and the LSD mixed checkpoint re-evaluates to pass@1
0.015, sampled reward 0.0039 — confirming pilots 1–3's conclusion
(mixed SFT does not learn) with correct numbers. The
teacher-forced diagnostics in pilots 1–3 were unaffected (they use
correctly built sequences, not `generate`).

### Grid run 1 — VM restart (2026-11-03, ~01:48 ET)

The full grid (28 runs x 300 steps, B=12) was launched at ~01:35 ET
and ran at ~0.50 s/step — 4 runs completed (GRPO seeds 0–2, GSPO
seed 0) — then the VM restarted (uptime 1 min at 01:48 ET),
wiping /tmp and killing the grid process. The workspace persisted,
including the partial `results/experiment.json`. The venv and the
SFT checkpoint had lived in /tmp, so both were lost; they now live
in `~/cspo-venv` and `~/.cache/cspo-sft-checkpoint.npz`, and the
grid was relaunched from scratch. The 4 completed runs reproduce
identically (everything is seeded), so no results were lost — only
~13 minutes of compute.

### Grid run 2 — full grid (2026-11-03, ~01:50–03:02 ET)

28 runs x 300 steps, B=12, G=8, E=2, in 4350 s total (~0.50
s/step). SFT warm-start reproduced exactly (probe rows identical to
Pilot 4; pass@1 0.335). Headline final pass@1 (mean over seeds):
GRPO 0.330, GSPO 0.298, FSPO 0.317, CSPO 0.323, unclipped 0.307,
cspo-len 0.327 — no method beats the 0.335 warm-start, and the
spread is within seed noise. Medium/hard stay 0.000 everywhere
(one medium problem correct in the whole grid, under K=1). The
warm-start policy is nearly deterministic (mean rollout entropy
0.100 for every method), so within-group advantage variance is
thin. The CSPO diagnostics are the real result: realised
miscoverage 0.220 vs target α = 0.1; the band contracts from
(0.8, 1.2) to ~[0.99, 1.01] by step 50 and stays there (mean width
0.021). Larger α narrows the band and raises miscoverage
(α = 0.05/0.1/0.2/0.4 -> width 0.039/0.021/0.008/0.002,
miscoverage 0.120/0.220/0.417/0.801, saturation
0.054/0.101/0.177/0.271) — the W = 1 point mass (half of each
retained block is k=0 scores, exactly 1.0) pulls the α/2 quantile
toward 1 as α grows. cspo-len per-bucket miscoverage
0.252/0.257/0.298 vs CSPO marginal 0.220: the extension does not
improve length-conditional calibration here. Prop. 3.3 and Lemma
3.2 hold on 100% of steps for every sequence method. K = 1 is
nominally best on pass@1 (0.345) vs K = 2 (0.323) and K = 4
(0.325), within noise. All numbers in `results/measured.json`,
cross-footed to the README. Determinism: a rerun of CSPO seed 0
from the saved SFT checkpoint reproduces the grid run's
diagnostics and evals exactly (the only apparent mismatch was the
final band comparing as tuple vs JSON list — same values).
`results/*.json` use Python's `Infinity` token for the unclipped
method's unbounded band width/endpoint; the repo's tooling
(Python `json`) round-trips it, and `scripts/crossfoot.py`
compares infinities directly.

Pilot 1 CSPO run: 100 steps took 135.8 s -> 1.358 s/step
(B=16, G=8, E=2, d=64, float64). LSD pilot with the k=0 forward
reuse: 60 steps took 75.3 s -> 1.255 s/step. The full grid as first
sketched (28 runs x 400 steps = 11,200 steps) would take ~3.9 h
at that rate, so the grid is cut to 300 steps/run and B reduced
to 12 (the task's own fallback for a slow transformer), giving
28 runs x 300 steps = 8,400 steps at ~1.0 s/step, ~2.3 h. The
band behaved sanely in the pilots even with zero reward:
warm-start (0.8, 1.2) on step 0, then contracting to
~(0.96–1.00, 1.00–1.02) as the buffer filled with near-1 scores
(at update k=0 every score is exactly 1, so half of each retained
block is 1.0 by construction), with realised miscoverage at k>=1
between 0.05 and 0.14 against the 0.10 target — consistent with
Theorem B.2's `alpha + drift` shape.

## Bugs found and fixed

1. **Attention key-pad mask shape** (`policy.py`): the PAD mask was
   built as `(tokens == PAD)[None, None, None, :]`, giving shape
   `(1,1,1,N,T)` — a broadcast error on the first padded batch. Fixed
   to `(tokens == PAD)[:, None, None, :]` -> `(N,1,1,T)`. Caught by
   the first test run (12 failures).
2. **GRPO objective contaminated by non-response positions**
   (`objectives.py`, `train.py`): `_response_logp` returns 0 at
   non-response positions, so `exp(0-0) = 1` there, and the token-mean
   objective summed `(T - L_i)` extra copies of `A_i` per response.
   Fixed with an explicit response-position mask passed to
   `grpo_objective` / `grpo_coeffs`. Caught by code review before the
   first grid run; the brute-force test now covers it.
3. **Prop. 3.3 bound for per-response bands** (`objectives.py`): the
   lower-tail term `a * p-` assumed a scalar band; FSPO and cspo-len
   use per-response bands, making the bound an array. Generalised to
   `mean(a_i * 1{W_i < a_i})`, which reduces to `a * p-` for a scalar
   band. Caught by the first test run.
4. **`parse_response` intermediate extraction** (`tasks.py`): the
   segment before each `=` is `<r_i> <op> <operand>`, not the
   intermediate alone — the intermediate is its leading digit run.
   Caught by the task tests.
5. **Test sign error** (`tests/test_policy.py`): the finite-difference
   gradient checks passed `-coeffs` to `_d_logits_from_coeffs`, which
   already negates internally (it takes `dJ/d log p` and returns the
   loss gradient). Fixed to pass `coeffs`. The checks now agree with
   finite differences to < 1e-5.
6. **`generate` inserted PADs between prompt and response**
   (`policy.py`): short prompts were right-padded up to the longest
   prompt, so generated tokens started after a run of PADs at
   unfamiliar absolute positions and greedy decoding derailed on
   mixed-length batches (full-test-set eval 0.0 vs easy-subset 0.84
   for the same model). Fixed by tracking per-row current lengths
   and right-padding only after each row's last real token.
   Caught by re-evaluating a saved checkpoint on the full test set.
   All mixed-tier greedy/sampled numbers before the fix were
   discarded and re-measured; see Pilot 6.
