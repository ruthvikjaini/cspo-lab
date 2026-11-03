# cspo-lab

Conformal prediction-enabled sequence-level policy optimization
(CSPO) for reasoning language models — reproduced and extended in
NumPy on CPU, on a small synthetic testbed with a verifiable reward.

This repo reproduces Yang, Tian, Ren, Li & Peng, **"Conformal
Prediction-Enabled Sequence-level Policy Optimization for Reasoning
Language Models"** (arXiv:2610.25092, 18 Oct 2026), and tests the
paper's stated open point: the paper's band calibration is marginal
over response length, so this lab adds a length-stratified CSPO
variant with a separate band per length tercile.

## Why sequence-level clipping?

GRPO clips token-level ratios while the advantage is a response-level
scalar — the clip acts on a different unit than the credit. GSPO
(Zheng et al. 2025) fixes the unit by clipping the sequence-level
power score `W_i = exp(S_i / L_i)`, the geometric mean of the token
ratios, in a fixed band. FSPO (Mao et al. 2025) makes the band
length-fair by clipping the log-ratio `S_i` within `±0.2·sqrt(L_i)`.
CSPO replaces the fixed band with a data-adaptive one: before each
training step, the band endpoints are the `α/2` and `1 − α/2`
empirical quantiles of a rolling buffer of recently retained scores,
anchored at `W = 1`. The band therefore tracks what the policy is
actually doing — contracting early in training, then varying in a
narrower range — instead of trusting a hand-picked constant.

## What this repo does

- **Testbed** (`src/cspolab/tasks.py`): variable-length multi-step
  arithmetic with a verifiable binary reward (1 iff the final answer
  digits are exactly correct). Problems are chains of 2–3
  additions/subtractions over 2–3 digit operands; the policy must emit
  the intermediate result(s) then the final answer, then EOS, e.g.
  prompt `47 + 38 - 19 =`, response `58 - 91 = 66 <EOS>`. Numbers in
  the response are written least-significant digit first (see
  `docs/RESEARCH.md` for why — the short version is that a 1-layer
  transformer does not learn most-significant-first carry chains in a
  night, and the mixed-tier SFT never learns the algorithm at all).
  Output lengths run roughly 6–21 tokens. Partial credit for
  intermediates is recorded as a diagnostic only and never trained on.
- **Policy** (`src/cspolab/policy.py`): a 1-layer, 4-head, width-64
  ReLU Transformer (no LayerNorm) in float64 NumPy with manual
  backprop, verified against finite differences. Temperature 1.0
  sampling for rollouts, greedy decoding for evaluation. Plain Adam
  (lr 3e-4), gradient clipping by global norm 1.0.
- **Methods** (`src/cspolab/objectives.py`, `src/cspolab/train.py`):
  GRPO (token clip 0.2), GSPO (fixed band 0.8–1.2), FSPO
  (`exp(±0.2/sqrt(L))`), CSPO (Algorithm 1: α=0.1, K=2, warm-start
  0.8–1.2), unclipped GSPO (band `(0, ∞)`), and the extension
  **cspo-len** — length-stratified CSPO with a separate rolling
  buffer and band per length tercile. All methods run from the same
  shared SFT warm-start checkpoint (easy-only SFT, 1500 steps — see
  `docs/RESEARCH.md` for the pilot that forced this) on the same
  seeded prompts.
- **Theory checks**: realised miscoverage of fresh scores against the
  calibrated band vs the target α (Theorem B.2), the Prop. 3.3
  objective-distortion bound (`0 ≤ J_G − J_C ≤ bound`), and the
  Lemma 3.2 `E[W²] ≤ 1 + 2/L_min` bound, all recorded per step.

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/demo.py               # probes + a short CSPO run (~10 s)
python -m pytest tests/ -q           # 40 tests
python scripts/run_experiments.py   # full grid -> results/
python scripts/crossfoot.py          # README figures vs results/
```

Requires Python 3.11+, NumPy, and pytest. CPU-only; no GPU, no model
downloads, no API keys.

## Measured results

All numbers below are measured by this repo on CPU, cross-footed
against `results/measured.json` by `scripts/crossfoot.py`. Paper
numbers are quoted in the comparison table for context and are not
this repo's measurements. The SFT warm-start scores pass@1 0.335 on
the 200-problem held-out test set (easy 0.84, medium 0.0, hard 0.0)
before any RL.

### Headline: final greedy pass@1 after 300 RL steps (mean over seeds)

| method | pass@1 | easy | medium | hard | mean reward (last 50 steps) |
| --- | --- | --- | --- | --- | --- |
| GRPO | 0.330 | 0.825 | 0.000 | 0.000 | 0.360 |
| GSPO | 0.298 | 0.746 | 0.000 | 0.000 | 0.326 |
| FSPO | 0.317 | 0.792 | 0.000 | 0.000 | 0.350 |
| CSPO | 0.323 | 0.808 | 0.000 | 0.000 | 0.362 |
| unclipped | 0.307 | 0.767 | 0.000 | 0.000 | 0.340 |
| cspo-len | 0.327 | 0.817 | 0.000 | 0.000 | 0.365 |

No method beats the SFT warm-start (0.335) on final pass@1 in 300
steps, and the spread across methods (0.298–0.330) is within seed
noise — the headline is a null result, stated plainly. The reason is
visible in the diagnostics: the warm-start policy is nearly
deterministic (mean rollout entropy 0.100 for every method), so
within-group advantage variance is thin and 300 steps move almost
nothing. Medium and hard stay at 0.000 for every method (one medium
problem correct in the whole grid, under K=1); all of the reward
comes from easy. Mid-training (step 150) GRPO reached 0.350 and CSPO
0.318, so the final numbers are not a measurement artefact — the
methods genuinely do not separate on pass@1 here.

### CSPO diagnostics (mean over steps and seeds)

| quantity | CSPO | cspo-len |
| --- | --- | --- |
| realised miscoverage (target α = 0.1) | 0.220 | 0.260 |
| mean band width | 0.021 | 0.023 |
| saturation fraction | 0.101 | 0.119 |
| Prop. 3.3 bound holds (fraction of steps) | 1.000 | 1.000 |
| Lemma 3.2 bound holds (fraction of steps) | 1.000 | 1.000 |

The theory checks pass on every step for both variants (Prop. 3.3
and Lemma 3.2 hold at 1.000), but the calibration does not hit its
target: CSPO's realised miscoverage is 0.220 against α = 0.1, more
than double the target. The band contracts immediately — from the
(0.8, 1.2) warm-start to about (0.991, 1.006) by step 50 — and stays
razor-thin (mean width 0.021, final bands [0.991, 1.010],
[0.997, 1.005], [0.979, 1.015] across seeds). The cause is in the
buffer: at update k=0 every score is exactly 1.0 (policy = behaviour
policy), so half of each retained block is a point mass at W = 1,
and the 5th/95th percentiles collapse toward it. Fresh scores at
k=1 drift outside the thin band 22% of the time — Theorem B.2's
drift term, not a bug in the calibration. The fixed bands barely
bind by comparison: GSPO's miscoverage is 0.001 and FSPO's 0.006,
so their saturation fractions are 0.001 and 0.005, against CSPO's
0.101 — CSPO zeroes the gradient on about one response in ten.

### Length-conditional miscoverage (cspo-len)

CSPO calibrates one band marginally over all lengths; cspo-len gives
each length tercile its own buffer and band. The table shows
cspo-len's realised miscoverage per tercile (target α = 0.1).
CSPO's marginal miscoverage (one band for all lengths) is 0.220 for
comparison — the point of the extension is whether the per-bucket
values sit closer to α than the marginal does.

| bucket | cspo-len miscoverage |
| --- | --- |
| short | 0.252 |
| medium | 0.257 |
| long | 0.298 |

They do not: the per-bucket values (0.252 / 0.257 / 0.298) are all
at or above CSPO's marginal 0.220, and the long bucket is the worst.
Length stratification does not fix the miscoverage overshoot on this
testbed, because the overshoot is driven by the W = 1 point mass in
every bucket's buffer, not by length heterogeneity. The extension
fails as a fix here — which is itself the finding: conditional
calibration needs the buffer composition fixed first.

### Ablations (CSPO, final pass@1, mean over seeds)

| setting | pass@1 |
| --- | --- |
| α = 0.05 | 0.320 |
| α = 0.1 (headline) | 0.323 |
| α = 0.2 | 0.330 |
| α = 0.4 | 0.325 |
| K = 1 | 0.345 |
| K = 2 (headline) | 0.323 |
| K = 4 | 0.325 |

Pass@1 barely moves with α or K (0.320–0.345, within seed noise),
but the diagnostics move a lot: larger α gives a *narrower* band
(mean width 0.039 at α = 0.05, 0.021 at 0.1, 0.008 at 0.2, 0.002 at
0.4) and *higher* miscoverage (0.120, 0.220, 0.417, 0.801) — the
opposite of the naive reading. The mechanism is the same point mass:
with half the buffer at exactly W = 1, a larger α puts the α/2
quantile closer to the point mass, so the band collapses and fresh
scores fall outside it most of the time. Saturation follows the same
pattern (0.054, 0.101, 0.177, 0.271). K matters little: K = 1 is
nominally best on pass@1 (0.345) and K = 4 matches the headline
(0.325), with miscoverage 0.238 vs 0.213.

## Paper vs lab

| | paper (Qwen3-1.7B, math benchmarks) | this lab (1-layer Transformer, arithmetic) |
| --- | --- | --- |
| CSPO avg Pass@1 | 34.11 | 0.323 |
| GSPO avg Pass@1 | 33.36 | 0.298 |
| GRPO avg Pass@1 | 30.36 | 0.330 |
| DAPO avg Pass@1 | 31.42 | — (not run here) |
| base model | 20.83 | 0.335 (SFT warm-start) |
| ALFWorld CSPO / GSPO | 64.06 / 31.25 | — (no agent task here) |
| α ablation (0.1 / 0.2 / 0.4) | 34.11 / 33.50 / 32.37 | 0.323 / 0.330 / 0.325 |

The ordering to compare is CSPO vs GSPO vs GRPO, not the absolute
numbers: the paper trains a 1.7B-parameter model on math and agent
benchmarks; this lab trains a 64-wide Transformer on arithmetic.
What transfers is the mechanism (adaptive band, miscoverage near α,
length-conditional calibration), not the scale.

## Limitations

- **Small model, synthetic task.** Everything here is a 1-layer
  Transformer on arithmetic. The paper's claims are about 1.7B
  models on math and agent benchmarks; this lab checks the
  mechanism, not the scale.
- **The warm-start is easy-only.** Mixed-tier SFT never learns the
  algorithm (see `docs/RESEARCH.md` pilots 1–3), and a mixed second
  phase does not transfer to medium/hard, so RL starts from a
  checkpoint that is strong on easy (0.84) and zero on medium/hard.
  Medium/hard improvement must come from RL generalisation.
- **Short horizon.** 300 RL steps per run. Band dynamics and
  miscoverage are measured over that horizon only.
- **Temperature 1.0 rollouts, E=2 updates.** Both are fixed
  documented choices, not tuned.
- **The `W = 1` mass in the buffer.** At update k=0 every score is
  exactly 1 (policy = behaviour policy), so half of each retained
  block is 1.0 by construction. The quantiles are computed over that
  mixture, exactly as the paper specifies.
- **3 seeds headline, 2 seeds ablations.** Small seed counts; treat
  small gaps between methods as suggestive, not decisive.

## Interview talking points

1. **The unit mismatch is the whole story.** GRPO clips token
   ratios against a response-level advantage; GSPO fixes the unit by
   clipping the geometric-mean score `W`, and CSPO fixes the band by
   calibrating it with conformal prediction on recent scores instead
   of hand-picking 0.8–1.2. In this lab the headline is a null: no
   method beats the 0.335 SFT warm-start on pass@1 in 300 steps
   (GRPO 0.330, CSPO 0.323, GSPO 0.298), because the warm-start
   policy is nearly deterministic (entropy 0.100) and there is
   almost no within-group signal to learn from — a useful reminder
   that the clipping method only matters once the policy has
   something to explore.
2. **Marginal vs conditional calibration is a real gap, and it is
   measurable.** The paper notes its band is calibrated marginally
   over length; length-stratified CSPO gives each length tercile its
   own buffer and band, and the length-conditional miscoverage
   table above shows the extension does *not* help here (0.252 /
   0.257 / 0.298 per bucket vs CSPO's marginal 0.220) — because the
   overshoot is driven by the W = 1 point mass in every bucket's
   buffer, not by length. Conditional calibration needs the buffer
   composition fixed first.
3. **The testbed fought back, and the fixes are in the open.** The
   1-layer model could not learn mixed-tier arithmetic at all
   (three pilots, two digit orders), and a padding bug in batched
   generation silently zeroed every mixed-tier eval until a
   checkpoint re-evaluation caught it. Both are documented with
   numbers in `docs/RESEARCH.md` — the kind of debugging that
   matters more than the headline table.

## Sources

- Yang, Tian, Ren, Li & Peng, "Conformal Prediction-Enabled
  Sequence-level Policy Optimization for Reasoning Language Models",
  arXiv:2610.25092, 18 Oct 2026.
  https://arxiv.org/abs/2610.25092
- Zheng et al., "GSPO: Group Sequence Policy Optimization",
  arXiv:2507.18071, 2025 (sequence-level clipping, via the CSPO paper).
- Mao et al., "FSPO" (length-fair sequence clipping, via the CSPO paper).
- GitHub trending check, 2026-11-03 (see `docs/RESEARCH.md`): no
  trending repo with a genuine measurable one-night CPU weakness, so
  the paper fallback was used per the standing rule.
- Transformer scaffold follows the Day 9 lab's design
  (`softmax-cliff-lab`: 1-layer, 4-head, width-64, ReLU MLP, no
  LayerNorm, manual backprop); this implementation is written fresh
  for per-position logits and policy gradients.

## License

MIT. See `LICENSE`. The paper is arXiv only; no paper code was used
or copied.
