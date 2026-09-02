# NanoGPT: A Transformer From Scratch

*A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare.*

**Task** Self-supervised sequence modelling  ·  **Domain** Deep learning / NLP  ·  **Primary metric** Validation cross-entropy (bits/char)

> Generated from the training run of `2026-09-02T20:21:34+00:00` (commit `n/a`, seed 42). Every figure below is read from that run's artifacts.

## Abstract

A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare. The system follows the full CRISP-DM cycle on 1,115,394 rows of Tiny Shakespeare character corpus, selecting NanoGPT by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: parameters 812,288, val bits per char 2.1714, perplexity 4.505, beats unigram baseline yes. Best validation loss 1.5051 nats (2.1714 bits/char, perplexity 4.505). That is 2.608 bits below the unigram baseline and 3.851 below uniform. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

## 1. Business understanding

**Question.** Can a transformer small enough to train on a laptop in minutes learn the statistical structure of English text -- and can we demonstrate that with a number rather than by showing off a paragraph of output?

**Success criteria, fixed before modelling:**

- Validation cross-entropy well below the unigram baseline of 4.7794 bits/char.
- Train and validation loss logged together so overfitting is locatable.
- Validation text held out as a contiguous block, never shuffled.
- Every architectural component implemented from primitives and explained.

**Why Validation cross-entropy (bits/char).** Held-out perplexity is the only honest measure for a generative model; sampled text is a demo, not an evaluation.

## 2. Data

- **Source** — Tiny Shakespeare character corpus
- **Rows** — 1,115,394 in the curated file; 1,115,394 after preparation
- **Licence** — Public domain

The curation rule and a SHA-256 for this file are recorded in [`data/MANIFEST.json`](../../../data/MANIFEST.json). Ingestion applies only column selection, dtype coercion, deterministic subsampling and stable sorting — no imputation, scaling or target-aware filtering.

## 3. Data Understanding

1,115,394 characters, 65 distinct symbols, 40,001 lines. The unigram distribution alone reaches 4.7794 bits/char against a uniform 6.0224.

**Findings.**

- *Character frequencies are far from uniform* — unigram entropy 4.7794 vs uniform 6.0224 bits. 1.24 bits are available from marginal frequencies alone, before any context is modelled.
- *The corpus has strong repeated structure* — speaker names and line breaks recur constantly. Easy structure a model learns first; the attention maps show exactly this.

**Decisions.**

- **Character-level rather than subword tokenisation.** A 65-symbol vocabulary can be printed in full, making the input representation completely transparent. *Rejected:* Byte-pair encoding — Better compression, but the tokeniser becomes an opaque component in a project whose purpose is transparency.

## 4. Data Preparation

1,003,855 training tokens and 111,539 validation tokens, split by position at character 1,003,855.

**Findings.**

- *The split is contiguous and never shuffled* — The final 10% of the text, as one contiguous block, is held out. Never shuffled: a random split would place a line in training and the next line in va. A shuffled split would score the model on completing passages it had effectively read.

**Decisions.**

- **Hold out the last 10% as a single block.** Guarantees the validation text is unseen continuation. *Rejected:* Random window sampling across the whole text — Adjacent windows overlap heavily; validation loss would measure memorisation.

## 5. Modelling

812,288 parameters: 4 pre-norm blocks, 4 heads, 128-dim residual stream, 128-token context. RoPE, RMSNorm, SwiGLU, tied embeddings.

**Leaderboard** — scored by validation bits per character (lower is better).

| Model | Bits/char |
|---|---|
| Uniform random | 6.0224 |
| Unigram frequency | 4.7794 |
| NanoGPT (4L/4H/128D) | 2.1714 |

> All three scored on the identical held-out final 10% of the corpus, which was never shuffled into training.

## 6. Evaluation

Best validation loss 1.5051 nats (2.1714 bits/char, perplexity 4.505). That is 2.608 bits below the unigram baseline and 3.851 below uniform.

**Headline results**

| Measure | Value |
|---|---|
| Parameters | 812,288 |
| Val bits per char | 2.1714 |
| Perplexity | 4.505 |
| Beats unigram baseline | yes |
| Improvement over unigram bits | 2.608 |
| Layers | 4 |
| Heads | 4 |
| Context | 128 |
| Training seconds | 249.4 |

## 7. Deployment

Weights and tokeniser are saved together; generation is served with live temperature and top-k controls at /api/projects/nanollm/generate.

Served at `POST /api/projects/nanollm/predict`. The fitted pipeline is pickled whole, so preprocessing travels with the model and training and serving cannot drift apart.

## 8. Limitations

- 812,288 parameters -- five orders of magnitude below a production model.
- 128-character context: it cannot track a sentence, let alone a scene.
- Character-level, so it must learn spelling from scratch and wastes capacity doing so.
- Single corpus, single style, no instruction following, no factual grounding.

**Out of scope**

- Answering questions. It has no knowledge, no instruction tuning and no notion of truth -- it models character sequences in one 1.1M-character corpus.
- Any production text generation.
- Generating text in any style other than the corpus it saw.

**Ethical considerations**

- A model this small cannot produce convincing misinformation, but the same architecture at scale can. The mechanism shown here -- next-token prediction with no notion of truth -- is exactly the mechanism behind confident fabrication in large language models.
- Trained on a single author's public-domain work; it reproduces that voice and nothing else, which makes it a poor demonstration of anything about language in general.

## 9. Audit

10 of 10 checks pass (grade A). *Scope:* Review for the failure modes specific to generative modelling: lookahead through a missing causal mask, shuffled text splits, and sample quality substituted for measurement.

| Check | Status | Evidence |
|---|---|---|
| Validation block contiguous and unshuffled | pass | The final 10% of the text, as one contiguous block, is held out. Never shuffled: a random split would place a line in training and the next line in validation, so the mod |
| Causal masking enforced | pass | Lower-triangular mask registered as a buffer inside attention; the published attention maps are exactly zero above the diagonal. |
| No batch spans the train/validation boundary | pass | get_batch samples windows from one split tensor at a time, and the two tensors are disjoint slices of the corpus. |
| Metric compared against explicit baselines | pass | Validation 2.1714 bits/char against a 4.7794 unigram floor and 6.0224 uniform. |
| Sample output not presented as evidence | pass | Samples are labelled illustrative; every claim rests on held-out cross-entropy. |
| Overfitting reported, not cropped | pass | Train and validation loss first separate by more than 0.05 nats at step 125. Beyond that the model is spending capacity memorising rather than general |
| Best checkpoint selected on validation, not training | pass | State dict snapshotted whenever validation loss improves. |
| Evaluation windows fixed across steps | pass | The loss estimator re-seeds its generator each call, so successive measurements are on identical windows and the curve is comparable. |
| Capability limits stated | pass | Model card states the model has no knowledge and cannot answer questions. |
| Determinism | pass | torch, numpy and the batch generator all seeded at 42. |

**Phase gates:** 6 of 6 passed.

- PASS — 1. Business Understanding
- PASS — 2. Data Understanding
- PASS — 3. Data Preparation
- PASS — 4. Modeling
- PASS — 5. Evaluation: 2.1714 bits/char against a 4.7794 unigram floor.
- PASS — 6. Deployment

## 10. Where this project looped back

CRISP-DM is iterative. These are the points where a later phase sent the work back to an earlier one — normally the part deleted before publication.

- The first architecture used learned absolute position embeddings and LayerNorm. Switching to RoPE and RMSNorm improved validation loss at identical parameter count and removed a whole embedding table.
- An early run reported an implausibly low training loss. The cause was a missing causal mask in the attention block -- the model was reading ahead. The test suite now asserts that attention above the diagonal is exactly zero, which is the generative equivalent of a leakage check.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_all.py --only nanollm
```

Every estimator, split and sampler is seeded, so a rerun on the same data reproduces this leaderboard exactly.
