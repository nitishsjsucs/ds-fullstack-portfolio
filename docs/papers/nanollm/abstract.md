# NanoGPT: A Transformer From Scratch — Abstract

**Task** Self-supervised sequence modelling  ·  **Data** Tiny Shakespeare character corpus  ·  **Metric** Validation cross-entropy (bits/char)

A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare. The system follows the full CRISP-DM cycle on 1,115,394 rows of Tiny Shakespeare character corpus, selecting NanoGPT by comparative evaluation on cross-validated scores with the hold-out reserved for a single final measurement. Headline results: parameters 812,288, val bits per char 2.1714, perplexity 4.505, beats unigram baseline yes. Best validation loss 1.5051 nats (2.1714 bits/char, perplexity 4.505). That is 2.608 bits below the unigram baseline and 3.851 below uniform. A static leakage audit of the training path passes 10 of 10 checks, and 6 of 6 CRISP-DM phase gates are met.

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

## What this model must not be used for

- Answering questions. It has no knowledge, no instruction tuning and no notion of truth -- it models character sequences in one 1.1M-character corpus.
- Any production text generation.
- Generating text in any style other than the corpus it saw.

## Known limitations

- 812,288 parameters -- five orders of magnitude below a production model.
- 128-character context: it cannot track a sentence, let alone a scene.
- Character-level, so it must learn spelling from scratch and wastes capacity doing so.
- Single corpus, single style, no instruction following, no factual grounding.

Full write-up: [`paper.md`](./paper.md).
