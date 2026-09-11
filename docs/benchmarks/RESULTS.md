# Benchmark interpretation

These release assets are derived from the final experiment records for *When Self-Consistency Hurts: The Hidden Cost of Majority Voting in Small Language Models*.

## Experimental grid

- Models: Qwen-2.5-0.5B-Instruct and Gemma-3-1B-it.
- Precision: MLX FP16 and affine weight-only MLX Q4 (group size 64).
- Tasks: GSM8K and MATH-500, first 200 prompts per cell.
- Decoding: one deterministic pass; five temperature-0.7, top-p-0.95 samples for the measured SC@5 outcome.
- Uncertainty: 1,000 prompt-level paired bootstrap resamples.
- Generation seeds: one seed per cell. The intervals do not include generation-seed variance.

## Pooled gate results

| Cell | Gate | Skip % [95% CI] | Cost | Retention [95% CI] |
|---|---|---:|---:|---:|
| Qwen-0.5B FP16 | logistic | 86.0 [82.5, 89.3] | 1.56× | 98.1 [82.5, 112.5] |
| Qwen-0.5B Q4 | confidence | 99.3 [98.5, 100.0] | 1.03× | 98.7 [76.5, 120.6] |
| Gemma-1B FP16 | logistic | 44.7 [40.0, 49.5] | 3.21× | 98.8 [86.2, 112.4] |
| Gemma-1B Q4 | logistic | 57.9 [53.2, 62.7] | 2.68× | 98.5 [77.9, 122.0] |

The skip-rate intervals are tight enough to support a compute-saving claim in this grid. The retention intervals are wide because the denominator—always-SC accuracy—is only 15–36%. Consequently, the README reports retention as point estimates and does not claim a tight, simultaneous compute-and-accuracy guarantee.

### Deployment accounting for the README headline

The selected Qwen-0.5B Q4 policy escalated 3 of 400 prompts. Always-SC@5 therefore requires 2,000 generation passes. The packaged implementation conservatively counts its routing draft as real compute and uses five fresh voters on escalation: `400 + 3 × 5 = 415` passes. This is 79.25% fewer generation passes than always-SC@5, reported as 79.2% in the README. Generation passes are a transparent compute proxy, not a claim about identical wall-clock, energy, or token cost per pass.

## Selection caveat

Scores are out-of-fold, but the operating threshold is selected on the same pooled scores used for reporting. The estimates can therefore be optimistic. The serialized logistic coefficients are refit on all 400 prompts after selecting the threshold. New deployments should use a separate threshold-selection split or nested cross-validation.

## Provenance

`per_task.csv` and `pooled_gate.csv` include SHA-256 pointers to the exact source files. `profile_manifest.json` records the source and artifact hashes for every bundled profile. Apart from the two attributed GSM8K demo questions, raw benchmark prompts are excluded; raw completion corpora and model weights are not distributed.
