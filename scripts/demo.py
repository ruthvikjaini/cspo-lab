"""Quick demo: exact probes + a tiny CSPO run, in under a minute.

Runs the no-training probes (band calibration, FSPO endpoints, the
Prop. 3.3 / Lemma 3.2 checks on synthetic scores) and a short
SFT + CSPO training run on a handful of problems, then prints the
headline numbers. No grid, no file writes.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cspolab import objectives as obj  # noqa: E402
from cspolab.evaluate import evaluate  # noqa: E402
from cspolab.policy import Transformer, copy_params  # noqa: E402
from cspolab.tasks import VOCAB_SIZE, make_test_set  # noqa: E402
from cspolab.train import run_training, sft_pretrain  # noqa: E402


def main():
    print("=== CSPO band calibration (synthetic buffer) ===")
    rng = np.random.default_rng(0)
    buffer = rng.normal(1.0, 0.15, size=2000)
    a, b = obj.calibrate_band(buffer, alpha=0.1)
    print(f"  alpha=0.1 buffer of 2000 scores -> band ({a:.3f}, {b:.3f})")
    fresh = rng.normal(1.0, 0.15, size=2000)
    print(f"  realised miscoverage on fresh scores: "
          f"{obj.miscoverage(fresh, a, b):.3f} (target 0.100)")
    print(f"  warm-start (empty buffer): "
          f"{obj.calibrate_band(np.array([]), alpha=0.1)}")

    print("\n=== FSPO length fairness ===")
    for length in (6, 9, 14, 21):
        fa, fb = obj.fspo_band([length])
        print(f"  L={length:>2}: W in ({fa[0]:.3f}, {fb[0]:.3f})")

    print("\n=== Prop. 3.3 / Lemma 3.2 on synthetic scores ===")
    w = rng.uniform(0.8, 1.25, size=64)
    adv = obj.group_advantages(rng.integers(0, 2, size=64).astype(float), 8)
    p33 = obj.prop33_check(w, adv, 0.95, 1.05, l_min=6, group_size=8)
    print(f"  gap {p33['gap']:.4f} <= bound {p33['bound']:.4f}: "
          f"{p33['holds']}")
    l32 = obj.lemma32_check(w, l_min=6)
    print(f"  E[W^2] {l32['e_w2']:.4f} <= {l32['bound']:.4f}: {l32['holds']}")

    print("\n=== Short SFT + CSPO run ===")
    t0 = time.time()
    model = Transformer(VOCAB_SIZE, seed=0)
    sft_pretrain(model, seed=0, steps=60, batch_size=16, log_every=30)
    test_set = make_test_set(40, seed=77)
    before = evaluate(model, test_set)["pass_at_1"]
    config = {"B": 4, "G": 4, "E": 2, "lr": 3e-4,
              "alpha": 0.1, "K": 2, "warm_c": 0.2, "eps": 0.2}
    result, _ = run_training("cspo", seed=0, steps=10,
                             init_params=copy_params(model.params),
                             test_set=test_set, config=config)
    after = result["evals"]["final"]["pass_at_1"]
    print(f"  SFT-only pass@1 {before:.3f} -> after 10 CSPO steps {after:.3f}")
    d = result["diagnostics"][-1]
    print(f"  last step: reward {d['reward']:.3f}, band "
          f"({d['band_a']:.3f}, {d['band_b']:.3f}), miscoverage "
          f"{d['miscoverage']}, sat frac {d['sat_frac']:.3f}")
    print(f"  ({time.time() - t0:.0f}s total; full grid: "
          f"python scripts/run_experiments.py)")


if __name__ == "__main__":
    main()
