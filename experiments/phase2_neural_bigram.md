# Phase 2 - Neural Bigram Language Model

## Reproducibility

- Git commit: `f8d7843` plus current working-tree changes
- Julia version: `1.12.5`
- Corpus: `data/raw/tiny_corpus.txt`
- Corpus SHA-256: `11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4`
- Vocabulary size: 47
- Seed: 42
- Configuration: `configs/neural_bigram.toml`

## Model

- Parameter matrix shape: `(vocab_size, vocab_size)`
- Orientation: `logits_table[next, current]`
- Parameter count for current corpus: `47 * 47 = 2209`
- Initialization: `Normal(0, 0.01)` with deterministic seed
- Objective: mean next-token cross-entropy
- Initial train NLL: `3.850648`
- Expected uniform NLL: `log(47) = 3.850148`

## Optimization

- Optimizer: SGD
- Learning rate: `0.1`
- Batch size: `64`
- Maximum steps: `2000`
- Best validation NLL observed: `2.576862`
- Loss implementation: stable log-sum-exp cross-entropy
- Training split only is used for parameter updates
- Validation split is used for model selection and diagnostics
- Test split is reserved for final reporting

## Required Sanity Checks

- Initial loss close to `log(vocab_size)`: pass.
- Gradient shape matches the parameter matrix: pass.
- Gradient values are finite and non-zero on a non-degenerate batch: pass.
- A single SGD step on a toy batch reduces loss: pass.
- Toy alternating corpus `[1, 2, 1, 2, ...]` overfits: pass.
- Generation is reproducible for the same seed: pass.
- Checkpoint save/load preserves logits and generated samples: pass.
- `Pkg.test()` passes Phase 0, Phase 1, and Phase 2 tests.

## Result Table

| Model | Train NLL | Validation NLL | Test NLL | Test PPL |
|---|---:|---:|---:|---:|
| Unigram count | 3.189685 | 3.167593 | 3.305356 | 27.258243 |
| Bigram count, alpha=0.1 | 1.621591 | 1.594022 | 1.750866 | 5.759588 |
| Neural bigram | 2.623950 | 2.576862 | 2.773992 | 16.022470 |

## Learning-Rate Sweep

| Learning rate | Best validation NLL | Best step | Stable |
|---:|---:|---:|---|
| 0.1 | 2.576862 | 2000 | yes |

The wider learning-rate sweep is still open experiment work. The implementation
now has the config/script/test support needed to run it reproducibly.

## Analysis Notes

- Neural bigram has the same one-token context limit as count bigram.
- Unlike unsmoothed count bigram, finite neural logits assign positive softmax probability to every next token.
- Neural bigram is not expected to always beat count bigram; the value of this phase is the differentiable training pipeline.
- In the first full-corpus run, neural bigram learned clear structure but did not match the smoothed count bigram baseline yet.
