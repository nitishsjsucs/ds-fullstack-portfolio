"""A decoder-only transformer, written out in full.

Every component is implemented from primitives rather than imported from
``torch.nn.Transformer``, because the point of this project is that the mechanism
should be legible. The architecture follows the modern Llama-style recipe rather
than the original 2017 paper:

* **RMSNorm** instead of LayerNorm -- no mean subtraction, no bias. Cheaper and
  empirically just as stable (Zhang & Sennrich 2019).
* **Pre-norm** residual blocks -- normalise the input to each sublayer rather than
  the output of the residual add. This is what makes deep stacks trainable
  without a learning-rate warmup babysitting act.
* **Rotary position embeddings** instead of learned absolute positions. RoPE
  encodes position by *rotating* the query and key vectors, so attention scores
  depend on relative offset. A model trained at 128 tokens degrades gracefully
  past that length instead of hitting a hard wall.
* **SwiGLU** feed-forward instead of ReLU MLP -- a gated activation that
  consistently outperforms at equal parameter count (Shazeer 2020).
* **Weight tying** between the token embedding and the output head, which removes
  a large parameter block and regularises a small model noticeably.

Causal masking is the load-bearing correctness property: position *t* may attend
to positions <= *t* and nothing later. Get that wrong and the model achieves a
spectacular training loss by reading the answer, which is the language-modelling
equivalent of target leakage. The test suite asserts it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return {**asdict(self), "head_dim": self.head_dim}


class RMSNorm(nn.Module):
    """Root-mean-square normalisation: rescale by RMS, no centring, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


def build_rope_cache(block_size: int, head_dim: int, theta: float,
                     device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary embeddings.

    Frequencies decay geometrically across the head dimension, so early pairs
    rotate quickly (encoding fine local position) and later pairs rotate slowly
    (encoding coarse global position).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    freqs = torch.outer(t, inv_freq)                    # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (q, k) pairs. x is (B, n_head, T, head_dim)."""
    T = x.size(-2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary positions."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly into n_head"
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # The causal mask. Registered as a buffer so it moves with the module and
        # is impossible to forget at inference time.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        weights = att if return_attention else None
        att = self.attn_dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return (y, weights) if return_attention else (y, None)


class SwiGLU(nn.Module):
    """Gated feed-forward: (SiLU(W1 x) * W3 x) W2."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # 8/3 rather than 4x: SwiGLU uses three matrices instead of two, and the
        # ratio keeps the parameter count comparable to a standard 4x ReLU MLP.
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)      # round up for kernel efficiency
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        a, weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, weights


class NanoGPT(nn.Module):
    """The full model: embed -> N blocks -> norm -> tied output head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output head reuses the embedding matrix. On a model
        # this small that block is a large fraction of all parameters, and tying
        # both shrinks it and acts as a regulariser.
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 recipe): keeps activation
        # variance stable as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attention: bool = False):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")

        x = self.drop(self.token_embedding(idx))
        attentions = []
        for block in self.blocks:
            x, w = block(x, return_attention=return_attention)
            if return_attention and w is not None:
                attentions.append(w)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return (logits, loss, attentions) if return_attention else (logits, loss)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, *,
                 temperature: float = 0.8, top_k: int | None = 40,
                 top_p: float | None = None) -> torch.Tensor:
        """Autoregressive sampling with temperature, top-k and nucleus filtering."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to the context window -- the model has no memory beyond it.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:                     # greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                continue

            logits = logits / temperature
            if top_k is not None and 0 < top_k < logits.size(-1):
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def architecture_summary(cfg: ModelConfig, model: NanoGPT) -> dict:
    """A component-by-component breakdown for the UI."""
    return {
        "config": cfg.to_dict(),
        "total_parameters": model.num_parameters(),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "components": [
            {"name": "Token embedding", "shape": f"{cfg.vocab_size} x {cfg.n_embd}",
             "params": cfg.vocab_size * cfg.n_embd,
             "role": "Maps each character to a learned vector.",
             "note": "Tied to the output head, so this block is counted once."},
            {"name": "Rotary position embedding", "shape": "cached, non-learned",
             "params": 0,
             "role": "Rotates queries and keys so attention scores depend on relative "
                     "offset rather than absolute index.",
             "note": "No parameters at all -- position is encoded geometrically."},
            {"name": f"{cfg.n_layer} x Causal self-attention",
             "shape": f"{cfg.n_head} heads x {cfg.head_dim} dims",
             "params": cfg.n_layer * (4 * cfg.n_embd * cfg.n_embd),
             "role": "Each position gathers information from earlier positions.",
             "note": "Lower-triangular mask enforces that nothing attends forward."},
            {"name": f"{cfg.n_layer} x SwiGLU feed-forward",
             "shape": f"{cfg.n_embd} -> {model.blocks[0].ffn.hidden} -> {cfg.n_embd}",
             "params": cfg.n_layer * (3 * cfg.n_embd * model.blocks[0].ffn.hidden),
             "role": "Per-position non-linear transformation.",
             "note": "Gated activation; ~8/3 expansion keeps parameters comparable to "
                     "a 4x ReLU MLP."},
            {"name": f"{2 * cfg.n_layer + 1} x RMSNorm", "shape": f"{cfg.n_embd}",
             "params": (2 * cfg.n_layer + 1) * cfg.n_embd,
             "role": "Stabilises activation scale before each sublayer.",
             "note": "Pre-norm placement is what makes the stack trainable without "
                     "learning-rate warmup gymnastics."},
            {"name": "Output head (tied)", "shape": f"{cfg.n_embd} x {cfg.vocab_size}",
             "params": 0,
             "role": "Projects back to vocabulary logits.",
             "note": "Shares weights with the token embedding."},
        ],
        "design_notes": [
            "RMSNorm over LayerNorm: no mean subtraction and no bias term.",
            "Pre-norm residuals: normalise the sublayer input, not the residual sum.",
            "RoPE over learned positions: relative offsets, graceful length extrapolation.",
            "SwiGLU over ReLU: gated activation, better quality at equal parameters.",
            "Weight tying: removes a whole vocab x embedding block and regularises.",
        ],
    }
