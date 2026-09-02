"""Character-level tokenisation and batching for the Shakespeare corpus.

The split is **by position, never shuffled**. For a language model, shuffling
would put text from the middle of a scene in training and its neighbouring lines
in validation -- the model then completes a passage it has effectively already
read, and validation loss stops measuring generalisation. Taking the last 10% as
one contiguous block means the validation text is genuinely unseen continuation.

Character-level rather than BPE is a deliberate teaching choice: the vocabulary is
65 symbols you can print in full, so "what the model sees" is not a black box, and
attention maps over characters are directly readable.
"""

from __future__ import annotations

import numpy as np
import torch

from ...core import paths


class CharTokenizer:
    """A bijection between the 65 characters in the corpus and integer ids."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        # Unknown characters are dropped rather than mapped to a sentinel: with a
        # 65-symbol vocabulary there is no meaningful <unk> to fall back on, and a
        # silent substitution would corrupt the context invisibly.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def describe(self) -> dict:
        printable = [repr(c)[1:-1] if c not in ("\n",) else "\\n" for c in self.chars]
        return {
            "vocab_size": self.vocab_size,
            "characters": printable,
            "note": ("Character-level: the entire vocabulary is 65 printable symbols, "
                     "so nothing about the input representation is hidden."),
        }


def load_text() -> str:
    return paths.curated("tiny_shakespeare").read_text(encoding="utf-8")


def build_dataset(val_fraction: float = 0.1) -> dict:
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n_val = int(len(data) * val_fraction)
    split_at = len(data) - n_val
    train_data, val_data = data[:split_at], data[split_at:]

    return {
        "tokenizer": tok,
        "train": train_data,
        "val": val_data,
        "meta": {
            "characters": len(text),
            "tokens": int(len(data)),
            "vocab_size": tok.vocab_size,
            "train_tokens": int(len(train_data)),
            "val_tokens": int(len(val_data)),
            "split_at_char": split_at,
            "split_policy": (
                f"The final {val_fraction:.0%} of the text, as one contiguous block, is "
                f"held out. Never shuffled: a random split would place a line in "
                f"training and the next line in validation, so the model would be "
                f"scored on completing a passage it had effectively already read."
            ),
            "line_count": text.count("\n") + 1,
        },
    }


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              generator: torch.Generator, device: str = "cpu"):
    """Sample random contiguous windows and their next-character targets.

    ``y`` is ``x`` shifted by one position, so predicting y[t] from x[:t+1] is the
    next-token objective. Both come from the same split, so no batch ever spans
    the train/validation boundary.
    """
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def character_statistics(text: str, tok: CharTokenizer, top: int = 20) -> dict:
    """Frequency profile and the entropy floor a unigram model would achieve."""
    ids = np.asarray(tok.encode(text))
    counts = np.bincount(ids, minlength=tok.vocab_size).astype(float)
    probs = counts / counts.sum()

    nz = probs[probs > 0]
    unigram_entropy = float(-(nz * np.log2(nz)).sum())

    order = np.argsort(-counts)[:top]
    return {
        "frequencies": [
            {"char": ("\\n" if tok.itos[int(i)] == "\n" else
                      "' '" if tok.itos[int(i)] == " " else tok.itos[int(i)]),
             "count": int(counts[i]), "probability": round(float(probs[i]), 5)}
            for i in order
        ],
        "unigram_entropy_bits": round(unigram_entropy, 4),
        "uniform_entropy_bits": round(float(np.log2(tok.vocab_size)), 4),
        "baselines": {
            "uniform_random": {
                "bits_per_char": round(float(np.log2(tok.vocab_size)), 4),
                "description": "Guess uniformly among 65 characters -- the absolute floor.",
            },
            "unigram_frequency": {
                "bits_per_char": round(unigram_entropy, 4),
                "description": ("Always predict the marginal character distribution, "
                                "ignoring context entirely. Any model that does not beat "
                                "this has learned nothing about language."),
            },
        },
        "note": ("Cross-entropy is reported in bits per character so it can be compared "
                 "directly against these two baselines. Nats would make the comparison "
                 "arithmetic rather than obvious."),
    }
