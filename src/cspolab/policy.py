"""Small autoregressive Transformer in NumPy, float64, manual backprop.

Architecture (following the Day 9 lab's scaffold: 1 layer, 4 heads,
width 64, ReLU MLP, no LayerNorm): 1 layer, 4 attention heads, width
``d_model=64``, ReLU MLP with hidden width 256, learned token and
positional embeddings, causal attention. Unlike the Day 9 lab, which
read out logits at the final position only, this model returns logits
at *every* position, because policy-gradient training needs the
log-probability of every generated token.

All parameters and all arithmetic are float64. That costs speed
against float32 but makes a full experiment grid byte-identical on
rerun (same shapes, same BLAS reduction order), which is what the
reproducibility checks need.

Padding: sequences in a batch are right-padded with PAD. Attention
masks PAD positions as keys, and the caller zeroes the loss gradient
at PAD positions. Positional embeddings are indexed by absolute
position.
"""

from __future__ import annotations

import numpy as np

from .tasks import EOS, PAD


def _xavier(rng, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class Transformer:
    """1-layer causal Transformer with per-position logits."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_mlp=256,
                 max_len=64, seed=0):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        self.max_len = max_len
        rng = np.random.default_rng(seed)
        emb_scale = 1.0 / np.sqrt(d_model)
        self.params = {
            "tok_emb": rng.normal(0, emb_scale, (vocab_size, d_model)),
            "pos_emb": rng.normal(0, emb_scale, (max_len, d_model)),
            "W_q": _xavier(rng, d_model, d_model),
            "W_k": _xavier(rng, d_model, d_model),
            "W_v": _xavier(rng, d_model, d_model),
            "W_o": _xavier(rng, d_model, d_model),
            "W_1": _xavier(rng, d_model, d_mlp),
            "b_1": np.zeros(d_mlp),
            "W_2": _xavier(rng, d_mlp, d_model),
            "b_2": np.zeros(d_model),
            "W_out": _xavier(rng, d_model, vocab_size),
            "b_out": np.zeros(vocab_size),
        }
        self.cache = None

    # -- forward ------------------------------------------------------
    def forward(self, tokens):
        """Return logits ``(N, T, vocab)`` for token ids ``(N, T)``."""
        p = self.params
        n, t = tokens.shape
        pos = np.arange(t)
        x = p["tok_emb"][tokens] + p["pos_emb"][None, pos, :]
        q = x @ p["W_q"]
        k = x @ p["W_k"]
        v = x @ p["W_v"]
        qh = q.reshape(n, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        kh = k.reshape(n, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        vh = v.reshape(n, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(self.d_head)
        scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale  # (N, H, T, T)
        causal = np.triu(np.ones((t, t), dtype=bool), k=1)
        key_pad = (tokens == PAD)[:, None, None, :]  # (N,1,1,T)
        mask = causal[None, None, :, :] | key_pad
        scores = np.where(mask, -np.inf, scores)
        s_max = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - s_max)
        attn = e / np.sum(e, axis=-1, keepdims=True)
        ctx = attn @ vh
        ctx = ctx.transpose(0, 2, 1, 3).reshape(n, t, self.d_model)
        attn_out = ctx @ p["W_o"]
        x1 = x + attn_out
        h_pre = x1 @ p["W_1"] + p["b_1"]
        h = np.maximum(h_pre, 0)
        mlp_out = h @ p["W_2"] + p["b_2"]
        x2 = x1 + mlp_out
        logits = x2 @ p["W_out"] + p["b_out"]
        self.cache = {
            "tokens": tokens, "x": x, "q": q, "k": k, "v": v,
            "qh": qh, "kh": kh, "vh": vh, "scores": scores,
            "attn": attn, "ctx": ctx, "attn_out": attn_out,
            "x1": x1, "h_pre": h_pre, "h": h, "mlp_out": mlp_out,
            "x2": x2,
        }
        return logits

    # -- backward -----------------------------------------------------
    def backward(self, d_logits):
        """Gradient of the loss w.r.t. every parameter.

        ``d_logits`` is the ``(N, T, vocab)`` gradient of the loss
        w.r.t. the output logits. Returns a dict of gradients with the
        same keys/shapes as ``self.params``.
        """
        p, c = self.params, self.cache
        n, t = c["tokens"].shape
        g = {}
        g["W_out"] = c["x2"].reshape(-1, self.d_model).T @ d_logits.reshape(-1, self.vocab_size)
        g["b_out"] = d_logits.sum(axis=(0, 1))
        dx2 = d_logits @ p["W_out"].T
        d_mlp_out = dx2
        g["W_2"] = c["h"].reshape(-1, self.d_mlp).T @ d_mlp_out.reshape(-1, self.d_model)
        g["b_2"] = d_mlp_out.sum(axis=(0, 1))
        dh = d_mlp_out @ p["W_2"].T
        dh = dh * (c["h_pre"] > 0)
        g["W_1"] = c["x1"].reshape(-1, self.d_model).T @ dh.reshape(-1, self.d_mlp)
        g["b_1"] = dh.sum(axis=(0, 1))
        dx1 = dx2 + dh @ p["W_1"].T
        d_attn_out = dx1
        g["W_o"] = c["ctx"].reshape(-1, self.d_model).T @ d_attn_out.reshape(-1, self.d_model)
        d_ctx = d_attn_out @ p["W_o"].T
        d_ctx_h = d_ctx.reshape(n, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        d_attn = d_ctx_h @ c["vh"].transpose(0, 1, 3, 2)
        d_vh = c["attn"].transpose(0, 1, 3, 2) @ d_ctx_h
        s = np.sum(d_attn * c["attn"], axis=-1, keepdims=True)
        d_scores = c["attn"] * (d_attn - s)
        causal = np.triu(np.ones((t, t), dtype=bool), k=1)
        key_pad = (c["tokens"] == PAD)[:, None, None, :]  # (N,1,1,T)
        mask = causal[None, None, :, :] | key_pad
        d_scores = np.where(mask, 0.0, d_scores)
        scale = 1.0 / np.sqrt(self.d_head)
        d_qh = (d_scores @ c["kh"]) * scale
        d_kh = (d_scores.transpose(0, 1, 3, 2) @ c["qh"]) * scale
        d_q = d_qh.transpose(0, 2, 1, 3).reshape(n, t, self.d_model)
        d_k = d_kh.transpose(0, 2, 1, 3).reshape(n, t, self.d_model)
        d_v = d_vh.transpose(0, 2, 1, 3).reshape(n, t, self.d_model)
        x_flat = c["x"].reshape(-1, self.d_model)
        g["W_q"] = x_flat.T @ d_q.reshape(-1, self.d_model)
        g["W_k"] = x_flat.T @ d_k.reshape(-1, self.d_model)
        g["W_v"] = x_flat.T @ d_v.reshape(-1, self.d_model)
        dx = (d_q @ p["W_q"].T + d_k @ p["W_k"].T + d_v @ p["W_v"].T
              + d_attn_out)
        flat_tokens = c["tokens"].ravel()
        one_hot = np.zeros((flat_tokens.shape[0], self.vocab_size))
        one_hot[np.arange(flat_tokens.shape[0]), flat_tokens] = 1.0
        g["tok_emb"] = one_hot.T @ dx.reshape(-1, self.d_model)
        g["pos_emb"] = np.zeros_like(p["pos_emb"])
        g["pos_emb"][:t] = dx.sum(axis=0)
        return g

    # -- sampling -----------------------------------------------------
    def log_softmax(self, logits):
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        return logits - m - np.log(e.sum(axis=-1, keepdims=True))

    def generate(self, prompts, rng=None, max_new_tokens=24, temperature=1.0,
                 greedy=False):
        """Sample (or greedily decode) a response for each prompt row.

        ``prompts`` is a list of token-id lists (variable length).
        Returns ``(responses, lengths)`` where ``responses`` is an
        ``(N, max_new_tokens)`` int array (PAD after EOS) and
        ``lengths`` counts tokens up to and including EOS (or the cap
        if EOS was never emitted).
        """
        n = len(prompts)
        prompt_len = max(len(pr) for pr in prompts)
        seqs = np.full((n, prompt_len + max_new_tokens), PAD, dtype=int)
        prompt_lens = np.array([len(pr) for pr in prompts])
        for i, pr in enumerate(prompts):
            seqs[i, :len(pr)] = pr
        finished = np.zeros(n, dtype=bool)
        lengths = np.full(n, max_new_tokens, dtype=int)
        for step in range(max_new_tokens):
            # Each row's current length is its own prompt length plus
            # the tokens generated so far. Forward on the batch cut to
            # the longest current row; shorter rows are right-padded
            # *after* their last real token (PAD keys are masked), so
            # every row keeps the training layout: prompt, then the
            # response immediately after it, with no gap. (An earlier
            # version padded short prompts up to the longest prompt,
            # inserting PADs between prompt and response; greedy
            # decoding then derailed on mixed-length batches.)
            cur_lens = prompt_lens + step
            max_cur = int(cur_lens.max())
            logits = self.forward(seqs[:, :max_cur])
            gathered = np.stack([logits[i, cur_lens[i] - 1, :]
                                 for i in range(n)])
            last = self.log_softmax(gathered)
            if greedy:
                nxt = last.argmax(axis=1)
            else:
                scaled = last / temperature
                probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                probs /= probs.sum(axis=1, keepdims=True)
                u = rng.random(n)
                cum = np.cumsum(probs, axis=1)
                nxt = (cum < u[:, None]).sum(axis=1)
                nxt = np.minimum(nxt, self.vocab_size - 1)
            active = ~finished
            seqs[active, cur_lens[active]] = nxt[active]
            just_finished = active & (nxt == EOS)
            lengths[just_finished] = step + 1
            finished |= (nxt == EOS)
            if finished.all():
                break
        responses = np.full((n, max_new_tokens), PAD, dtype=int)
        for i in range(n):
            responses[i, :lengths[i]] = \
                seqs[i, prompt_lens[i]:prompt_lens[i] + lengths[i]]
        return responses, lengths


class Adam:
    """Plain Adam (no weight decay), as used for the RL updates."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.step = 0

    def update(self, grads):
        self.step += 1
        b1, b2 = self.beta1, self.beta2
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.step)
            v_hat = self.v[k] / (1 - b2 ** self.step)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grads(grads, max_norm=1.0):
    """Clip by global norm; returns the pre-clip norm."""
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / total
        for k in grads:
            grads[k] = grads[k] * scale
    return total


def copy_params(params):
    return {k: v.copy() for k, v in params.items()}
