"""Run the full experiment grid and write results/.

Grid (all runs seeded; the SFT warm-start checkpoint is shared by
every run, so methods differ only in the RL update):
  headline  -- GRPO, GSPO, FSPO, CSPO, unclipped x 3 seeds x STEPS steps
  alpha     -- CSPO alpha in {0.05, 0.2, 0.4} x 2 seeds x STEPS steps
  K         -- CSPO K in {1, 4} x 2 seeds x STEPS steps
  extension -- cspo-len x 3 seeds x STEPS steps

Writes results/experiment.json (per-run, per-step diagnostics) and
results/measured.json (headline aggregates). Both files are
byte-identical on rerun: no wall-clock or timestamp fields are stored
(run timing is printed to stdout only).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cspolab.evaluate import evaluate  # noqa: E402
from cspolab.policy import Transformer, copy_params  # noqa: E402
from cspolab.tasks import VOCAB_SIZE, make_test_set  # noqa: E402
from cspolab.train import run_training, sft_pretrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASE_CONFIG = {"B": 12, "G": 8, "E": 2, "lr": 3e-4,
               "alpha": 0.1, "K": 2, "warm_c": 0.2, "eps": 0.2}
SFT_PHASE_STEPS = 1500
SFT_SEED = 0


def _mean(xs):
    return float(np.mean(xs))


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fraction_true(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarise_run(result):
    """Aggregate one run's per-step diagnostics + evals."""
    diags = result["diagnostics"]
    final = result["evals"]["final"]
    mid = result["evals"]["mid"]
    tail = diags[-50:]
    return {
        "method": result.get("label", result["method"]),
        "seed": result["seed"],
        "steps": result["steps"],
        "config": result["config"],
        "final_pass_at_1": final["pass_at_1"],
        "final_pass_at_1_by_tier": final["pass_at_1_by_tier"],
        "mid_pass_at_1": mid["pass_at_1"],
        "final_mean_length": final["mean_length"],
        "final_mean_partial_credit": final["mean_partial_credit"],
        "final_band": result["final_band"],
        "mean_reward_last_50": _mean([d["reward"] for d in tail]),
        "mean_reward_all": _mean([d["reward"] for d in diags]),
        "mean_entropy_last_50": _mean([d["entropy"] for d in tail]),
        "mean_sat_frac": _mean([d["sat_frac"] for d in diags]),
        "mean_miscoverage": _mean_or_none([d["miscoverage"] for d in diags]),
        "miscoverage_by_bucket": (
            [_mean_or_none([d["miscoverage_by_bucket"][b]
                            for d in diags if d["miscoverage_by_bucket"]])
             for b in range(3)]
            if result["method"] == "cspo-len" else None),
        "mean_band_width": _mean([d["band_width"] for d in diags]),
        "mean_band_a": _mean([d["band_a"] for d in diags]),
        "mean_band_b": _mean([d["band_b"] for d in diags]),
        "prop33_holds_fraction": _fraction_true(
            [d["prop33_holds"] for d in diags]),
        "prop33_mean_gap": _mean_or_none([d["prop33_gap"] for d in diags]),
        "prop33_mean_bound": _mean_or_none(
            [d["prop33_bound"] for d in diags]),
        "lemma32_holds_fraction": _fraction_true(
            [d["lemma32_holds"] for d in diags]),
        "lemma32_mean_e_w2": _mean_or_none(
            [d["lemma32_e_w2"] for d in diags]),
        "lemma32_mean_bound": _mean_or_none(
            [d["lemma32_bound"] for d in diags]),
    }


def aggregate(runs):
    """Aggregate per-method means across seeds."""
    by_method = {}
    for run in runs:
        by_method.setdefault(run["method"], []).append(run)
    out = {}
    for method, rs in by_method.items():
        entry = {
            "seeds": [r["seed"] for r in rs],
            "n_seeds": len(rs),
            "final_pass_at_1_mean": _mean([r["final_pass_at_1"] for r in rs]),
            "final_pass_at_1_by_seed": [r["final_pass_at_1"] for r in rs],
            "final_pass_at_1_by_tier_mean": {
                tier: _mean([r["final_pass_at_1_by_tier"][tier]
                             for r in rs])
                for tier in ("easy", "medium", "hard")},
            "mid_pass_at_1_mean": _mean([r["mid_pass_at_1"] for r in rs]),
            "mean_reward_last_50": _mean(
                [r["mean_reward_last_50"] for r in rs]),
            "mean_sat_frac": _mean([r["mean_sat_frac"] for r in rs]),
            "mean_miscoverage": _mean_or_none(
                [r["mean_miscoverage"] for r in rs]),
            "mean_band_width": _mean([r["mean_band_width"] for r in rs]),
            "prop33_holds_fraction": _mean_or_none(
                [r["prop33_holds_fraction"] for r in rs]),
            "lemma32_holds_fraction": _mean_or_none(
                [r["lemma32_holds_fraction"] for r in rs]),
        }
        if rs[0]["miscoverage_by_bucket"] is not None:
            entry["miscoverage_by_bucket"] = [
                _mean_or_none([r["miscoverage_by_bucket"][b] for r in rs])
                for b in range(3)]
        if rs[0]["method"].startswith("cspo"):
            entry["final_band_by_seed"] = [r["final_band"] for r in rs]
        out[method] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--headline-seeds", type=int, default=3)
    parser.add_argument("--ablation-seeds", type=int, default=2)
    parser.add_argument("--skip-sft", action="store_true",
                        help="reuse an existing SFT checkpoint file")
    parser.add_argument("--sft-steps", type=int, default=SFT_PHASE_STEPS,
                        help="SFT steps per curriculum phase")
    args = parser.parse_args()

    t_all = time.time()
    test_set = make_test_set(200, seed=12345)

    # Shared SFT warm-start checkpoint (kept outside the repo tree so
    # it never ships, and outside /tmp so a VM restart does not lose
    # it; the grid can reuse it with --skip-sft).
    ckpt = Path.home() / ".cache" / "cspo-sft-checkpoint.npz"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_sft and ckpt.exists():
        data = np.load(ckpt)
        init_params = {k: data[k] for k in data.files}
        print(f"reusing SFT checkpoint {ckpt}")
    else:
        model = Transformer(VOCAB_SIZE, seed=SFT_SEED)
        # Warm-start (see docs/RESEARCH.md): easy-only SFT, 1500 steps.
        # Mixed-tier SFT does not learn the algorithm (pilots 1-3), and
        # a mixed second phase does not transfer (medium stays 0.0, easy
        # slips 0.84 -> 0.78), so the shipped warm-start is easy-only.
        # Every RL method starts from this same checkpoint.
        rows1 = sft_pretrain(model, seed=SFT_SEED,
                             steps=args.sft_steps, batch_size=128, lr=3e-3,
                             log_every=250, tier_weights={"easy": 1.0})
        print("SFT (easy-only) probe rows:")
        for r in rows1:
            print("  ", r)
        init_params = copy_params(model.params)
        np.savez(ckpt, **init_params)
        print("  greedy pass@1 after SFT:",
              evaluate(model, test_set)["pass_at_1"])

    plan = []
    for method in ("grpo", "gspo", "fspo", "cspo", "unclipped"):
        for seed in range(args.headline_seeds):
            plan.append((method, seed, dict(BASE_CONFIG)))
    for alpha in (0.05, 0.2, 0.4):
        for seed in range(args.ablation_seeds):
            cfg = dict(BASE_CONFIG, alpha=alpha)
            plan.append((f"cspo-alpha-{alpha}", seed, cfg))
    for k in (1, 4):
        for seed in range(args.ablation_seeds):
            cfg = dict(BASE_CONFIG, K=k)
            plan.append((f"cspo-K{k}", seed, cfg))
    for seed in range(args.headline_seeds):
        plan.append(("cspo-len", seed, dict(BASE_CONFIG)))

    runs = []
    for label, seed, cfg in plan:
        method = label if label in (
            "grpo", "gspo", "fspo", "cspo", "unclipped", "cspo-len") else "cspo"
        t0 = time.time()
        result, _ = run_training(method, seed=seed, steps=args.steps,
                                 init_params=init_params,
                                 test_set=test_set, config=cfg)
        result["label"] = label
        runs.append(result)
        # Checkpoint after every run, so a crash mid-grid still leaves
        # usable partial results.
        (RESULTS / "experiment.json").write_text(json.dumps({"runs": runs}))
        print(f"[{label} seed={seed}] {args.steps} steps in "
              f"{time.time() - t0:.0f}s; final pass@1 "
              f"{result['evals']['final']['pass_at_1']:.3f}; "
              f"mean reward last 50 "
              f"{np.mean([d['reward'] for d in result['diagnostics'][-50:]]):.3f}",
              flush=True)

    summaries = [summarise_run(r) for r in runs]
    measured = {
        "sft_steps": {"easy_only": args.sft_steps},
        "steps_per_run": args.steps,
        "base_config": BASE_CONFIG,
        "methods": aggregate(summaries),
        "runs": summaries,
    }
    # Keep full per-step diagnostics in experiment.json only.
    experiment = {"runs": runs}
    (RESULTS / "experiment.json").write_text(json.dumps(experiment))
    (RESULTS / "measured.json").write_text(json.dumps(measured, indent=2))
    print(f"wrote results/experiment.json and results/measured.json "
          f"in {time.time() - t_all:.0f}s total")


if __name__ == "__main__":
    main()
