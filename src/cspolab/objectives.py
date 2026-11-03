"""Sequence-level policy objectives: GRPO, GSPO, FSPO, CSPO.

Notation follows Yang et al. (arXiv:2610.25092). For a group of ``G``
responses to one prompt, with token-level ratios ``r_{i,l}`` against
the behaviour policy:

* GRPO clips each token ratio: ``min(r A, clip(r, 1-eps, 1+eps) A)``,
  averaged over tokens, with a response-level scalar advantage ``A_i``.
* GSPO (Zheng et al. 2025) uses the sequence log-ratio
  ``S_i = sum_l log r_{i,l}`` and the power score
  ``W_i = exp(S_i / L_i)`` (the geometric mean of the token ratios),
  with a fixed clip band ``(1 - eps_low, 1 + eps_high)``:
  ``J_seq = E[(1/G) sum_i min(W_i A_i, clip(W_i) A_i)]``.
* FSPO (Mao et al. 2025) clips the sequence log-ratio ``S_i`` with a
  band scaling as ``sqrt(L_i)``: ``|S_i| <= 0.2 sqrt(L_i)``, i.e.
  ``W_i`` in ``exp(+/- 0.2 / sqrt(L_i))`` -- length fairness.
* CSPO (this paper, Algorithm 1) replaces the fixed band with a
  data-adaptive band calibrated by conformal prediction on a rolling
  buffer of retained scores (see :func:`calibrate_band`).

Advantages are group-normalised with the population standard
deviation, ``A_i = (R_i - mean R) / (std R + 1e-8)``.

Gradient rule (paper Prop. 3.4 endpoint convention): for a response
with advantage ``A`` and score ``W`` under band ``(a, b)``, the
gradient flows through ``W * A`` unless the response is saturated --
``A > 0`` and ``W > b``, or ``A < 0`` and ``W < a`` -- in which case
the gradient is exactly zero. At exact equality (``W == a`` or
``W == b``) the unclipped derivative is used. The same rule applies
per token for GRPO. ``dW / d log p_l = W / L`` for sequence scores,
so each token of a response carries coefficient ``A * W / (L * G)``
when not saturated.
"""

from __future__ import annotations

import numpy as np

EPS_A = 1e-8


# --- advantages and scores -------------------------------------------
def group_advantages(rewards, group_size):
    """Group-normalised advantages (population std + 1e-8).

    ``rewards`` is a flat array of length ``n_groups * group_size``.
    Returns an array of the same shape.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)  # population std (ddof=0)
    return ((r - mean) / (std + EPS_A)).ravel()


def sequence_scores(logp, logp_old, lengths):
    """Sequence log-ratios ``S`` and power scores ``W = exp(S / L)``.

    ``logp`` / ``logp_old`` are ``(N, T)`` arrays of per-token
    log-probs at response positions (0 elsewhere); ``lengths`` is the
    per-response token count. Returns ``(S, W)`` of shape ``(N,)``.
    """
    s = (logp - logp_old).sum(axis=1)
    w = np.exp(s / np.asarray(lengths, dtype=np.float64))
    return s, w


# --- bands ------------------------------------------------------------
def fspo_band(lengths, c=0.2):
    """Per-response FSPO band: ``W`` in ``exp(+/- c / sqrt(L))``."""
    lengths = np.asarray(lengths, dtype=np.float64)
    half = c / np.sqrt(lengths)
    return np.exp(-half), np.exp(half)


class RollingBuffer:
    """Rolling buffer of retained scores from the last ``K`` blocks.

    A block is one completed training step's retained scores (the
    scores computed at each of its ``E`` updates -- repeat scoring of
    the same response counts as a separate entry, per the paper).
    Only the most recent ``K`` blocks are kept.
    """

    def __init__(self, k=2):
        self.k = k
        self.blocks = []

    def add_block(self, scores):
        self.blocks.append(np.asarray(scores, dtype=np.float64).ravel())
        self.blocks = self.blocks[-self.k:]

    def values(self):
        if not self.blocks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self.blocks)

    def __len__(self):
        return int(sum(b.size for b in self.blocks))


def empirical_quantile(values, q):
    """Inverse empirical CDF (order-statistic) quantile.

    Convention: ``np.quantile(values, q, method="lower")`` -- the
    smallest order statistic whose empirical CDF is >= ``q``. This is
    the convention the paper's empirical quantiles are implemented
    with here; it is documented because the band endpoints depend on
    it at small buffer sizes.
    """
    return float(np.quantile(values, q, method="lower"))


def calibrate_band(buffer_values, alpha=0.1, warm_c=0.2):
    """CSPO band from a rolling buffer (paper Algorithm 1, Eq. 7).

    ``alpha_lo = alpha_hi = alpha / 2``; ``q_lo`` is the
    ``alpha/2`` empirical quantile of the buffer and ``q_hi`` the
    ``1 - alpha/2`` quantile. Anchored at ``W = 1``:
    ``a = min(q_lo, 1)``, ``b = max(q_hi, 1)``. An empty buffer uses
    the warm-start band ``(1 - c, 1 + c)``.
    """
    if buffer_values is None or len(buffer_values) == 0:
        return 1.0 - warm_c, 1.0 + warm_c
    q_lo = empirical_quantile(buffer_values, alpha / 2.0)
    q_hi = empirical_quantile(buffer_values, 1.0 - alpha / 2.0)
    return min(q_lo, 1.0), max(q_hi, 1.0)


def length_buckets(lengths):
    """Tercile buckets (short / medium / long) by output length.

    Cutoffs are the 1/3 and 2/3 quantiles of the batch's own lengths
    (``method="lower"`` order statistics). Returns an int array of
    bucket ids 0, 1, 2.
    """
    lengths = np.asarray(lengths)
    lo = np.quantile(lengths, 1.0 / 3.0, method="lower")
    hi = np.quantile(lengths, 2.0 / 3.0, method="lower")
    return np.where(lengths <= lo, 0, np.where(lengths <= hi, 1, 2))


# --- objectives -------------------------------------------------------
def _saturated(advantages, scores, a, b):
    """Saturation mask: ``(A > 0 and W > b) or (A < 0 and W < a)``.

    ``a`` / ``b`` may be scalars or per-response arrays. Exact
    equality is NOT saturation (Prop. 3.4 endpoint convention).
    """
    return ((advantages > 0) & (scores > b)) | ((advantages < 0) & (scores < a))


def sequence_objective(w, advantages, a, b):
    """CSPO/GSPO/FSPO objective value ``(1/G) sum min(W A, clip(W) A)``."""
    clipped = np.clip(w, a, b)
    return float(np.mean(np.minimum(w * advantages, clipped * advantages)))


def unclipped_sequence_objective(w, advantages):
    """``J_G = (1/G) sum W A`` -- the unclipped sequence objective."""
    return float(np.mean(w * advantages))


def sequence_coeffs(w, advantages, a, b, lengths):
    """Per-token gradient coefficients for a sequence-level objective.

    Returns ``coeffs`` of shape ``(N, T)`` such that
    ``dJ / d log p_{i,l} = coeffs[i, l]`` (zero at non-response
    positions, which the caller fills in). ``w`` is the per-response
    power score, ``lengths`` the per-response token count. The
    ``1/G`` normalisation is included: ``J`` is a mean over the
    ``G`` responses of the group... note the caller passes the full
    batch, so the mean is over all ``N`` responses in the batch.
    """
    n = w.shape[0]
    active = ~_saturated(advantages, w, a, b)
    per_response = np.where(active, advantages * w, 0.0)
    per_response = per_response / (np.asarray(lengths, dtype=np.float64) * n)
    return per_response


def grpo_coeffs(ratios, advantages, eps, lengths, mask):
    """Per-token gradient coefficients for the GRPO token objective.

    ``ratios`` is ``(N, T)`` of token ratios at response positions
    (0 elsewhere); ``advantages`` is ``(N,)``; ``mask`` is a bool
    ``(N, T)`` array marking response positions. ``J`` is the mean
    over responses of the mean over tokens of
    ``min(r A, clip(r, 1-eps, 1+eps) A)``. Returns coefficients of
    shape ``(N, T)`` for ``dJ / d log p`` (zero at non-response
    positions).
    """
    n, t = ratios.shape
    a = 1.0 - eps
    b = 1.0 + eps
    sat = _saturated(advantages[:, None], ratios, a, b)
    active = ~sat
    coeff = np.where(active, ratios * advantages[:, None], 0.0)
    coeff = coeff / (np.asarray(lengths, dtype=np.float64)[:, None] * n)
    return coeff * mask


def grpo_objective(ratios, advantages, eps, lengths, mask):
    """GRPO objective value (mean over responses of token means)."""
    a, b = 1.0 - eps, 1.0 + eps
    clipped = np.clip(ratios, a, b)
    per_token = np.minimum(ratios * advantages[:, None],
                           clipped * advantages[:, None])
    per_token = per_token * mask
    lengths = np.asarray(lengths, dtype=np.float64)
    per_response = per_token.sum(axis=1) / lengths
    return float(per_response.mean())


# --- diagnostics: paper theory checks ---------------------------------
def prop33_check(w, advantages, a, b, l_min, group_size):
    """Proposition 3.3: ``0 <= J_G - J_C <= A_max [sqrt((1+2/L_min) p+) + a p-]``.

    ``A_max = sqrt(G)`` (population-std convention for
    group-normalised advantages), ``p+ = P(W > b)``,
    ``p- = P(W < a)`` empirical. Returns a dict with the empirical
    gap, the bound, and whether the bound holds.
    """
    j_g = unclipped_sequence_objective(w, advantages)
    j_c = sequence_objective(w, advantages, a, b)
    gap = j_g - j_c
    a_max = np.sqrt(float(group_size))
    p_plus = float(np.mean(w > b))
    p_minus = float(np.mean(w < a))
    # Per-response bands (FSPO, cspo-len): the lower-tail term is the
    # mean of a_i over responses below their own lower endpoint; for
    # a scalar band this reduces to a * p-.
    lower_term = float(np.mean(np.asarray(a, dtype=np.float64)
                               * (np.asarray(w) < np.asarray(a))))
    bound = a_max * (np.sqrt((1.0 + 2.0 / l_min) * p_plus) + lower_term)
    return {"gap": gap, "bound": bound, "holds": bool(gap <= bound + 1e-12),
            "p_plus": p_plus, "p_minus": p_minus}


def lemma32_check(w, l_min):
    """Lemma 3.2: ``E[W^2] <= 1 + 2 / L_min``.

    Returns a dict with the empirical mean of ``W^2``, the bound,
    and whether it holds.
    """
    e_w2 = float(np.mean(np.asarray(w, dtype=np.float64) ** 2))
    bound = 1.0 + 2.0 / l_min
    return {"e_w2": e_w2, "bound": bound,
            "holds": bool(e_w2 <= bound + 1e-12)}


def miscoverage(w, a, b):
    """Fraction of scores outside the band ``(a, b)`` (strict)."""
    w = np.asarray(w, dtype=np.float64)
    return float(np.mean((w < a) | (w > b)))


def saturation_fraction(w, advantages, a, b):
    """Fraction of responses whose gradient is clipped to zero."""
    return float(np.mean(_saturated(advantages, w, a, b)))
