# NanoGPT: A Transformer From Scratch

*A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare.*

## The question

Can a transformer small enough to train on a laptop in minutes learn the statistical structure of English text -- and can we demonstrate that with a number rather than by showing off a paragraph of output?

## Why this metric and not accuracy

Held-out perplexity is the only honest measure for a generative model; sampled text is a demo, not an evaluation.

## What the data actually looked like

1,115,394 characters, 65 distinct symbols, 40,001 lines. The unigram distribution alone reaches 4.7794 bits/char against a uniform 6.0224.

**Character frequencies are far from uniform** — unigram entropy 4.7794 vs uniform 6.0224 bits. 1.24 bits are available from marginal frequencies alone, before any context is modelled.

**The corpus has strong repeated structure** — speaker names and line breaks recur constantly. Easy structure a model learns first; the attention maps show exactly this.

## The modelling

812,288 parameters: 4 pre-norm blocks, 4 heads, 128-dim residual stream, 128-token context. RoPE, RMSNorm, SwiGLU, tied embeddings.

> **Keep the best-validation checkpoint, not the final one.** The final step is not necessarily the best step.
>
> *We rejected Ship the last checkpoint:* Discards the genuinely best model whenever overfitting has begun.

> **Implement every component from primitives.** The project's purpose is that the mechanism be legible.
>
> *We rejected torch.nn.TransformerDecoder:* Fewer lines, but the interesting parts become invisible.

## Results

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

## The finding worth reading twice

**Sample quality is not a metric**

Shakespeare-shaped text is achievable at any loss level. Samples are labelled as illustration; every claim rests on validation loss.

## What went wrong first

Every project here records where a later phase sent the work back to an earlier one. That is normally the part deleted before publication, and it is usually the most useful part.

- The first architecture used learned absolute position embeddings and LayerNorm. Switching to RoPE and RMSNorm improved validation loss at identical parameter count and removed a whole embedding table.
- An early run reported an implausibly low training loss. The cause was a missing causal mask in the attention block -- the model was reading ahead. The test suite now asserts that attention above the diagonal is exactly zero, which is the generative equivalent of a leakage check.

## What it cannot do

- Answering questions. It has no knowledge, no instruction tuning and no notion of truth -- it models character sequences in one 1.1M-character corpus.
- Any production text generation.
- Generating text in any style other than the corpus it saw.

## Try it

The live inference playground for this project is on the **Live inference** tab of the console, or call it directly:

```bash
curl -X POST localhost:8000/api/projects/nanollm/predict \
  -H 'Content-Type: application/json' -d '{}'
```
