<div align="center">

# SiftSC

### Think once. Vote only when it helps.

Selective self-consistency for small, local language models. No API key. No fine-tuning. No cloud inference.

[![CI](https://github.com/heywanrong/SiftSC/actions/workflows/ci.yml/badge.svg)](https://github.com/heywanrong/SiftSC/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-7c3aed)](LICENSE)

<a href="#run-it-now"><img alt="Run the live demo" src="https://img.shields.io/badge/%E2%96%B6_RUN_THE_LIVE_DEMO-111827?style=for-the-badge"></a>
&nbsp;
<a href="#chat-with-the-model"><img alt="Start local chat" src="https://img.shields.io/badge/START_LOCAL_CHAT-0891b2?style=for-the-badge"></a>

![SiftSC uses fewer generation passes while retaining self-consistency accuracy](docs/assets/hero.svg)

</div>

Small models do not need five opinions for every question. SiftSC uses dramatically less compute while keeping performance close to Always-SC—and avoids some of the cases where majority voting makes a correct small-model answer worse. It first asks for one answer, then uses a tiny router to decide whether that answer should stand or whether five independent reasoning traces should vote.

## Run it now

On an Apple-Silicon Mac with Python 3.11+, one command installs SiftSC, downloads the public 4-bit model from Hugging Face, and runs a verified comparison:

```bash
python -m pip install "siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git" && siftsc demo
```

The first run downloads [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3) (about 290 MB). SiftSC pins the tested model revision and caches it for every run after that.

### One model. Two jobs: repair and protect.

When voting helps, SiftSC escalates:

```text
Question  Henry made two stops during his 60-mile bike trip. He first
          stopped after 20 miles. His second stop was 15 miles before
          the end. How far did he travel between the two stops?

PLAIN    15  ✗   (1 deterministic pass)
SIFTSC   25  ✓   (votes: 20 · 15 · 25 · 25 · 30)
```

The second stop is at mile `60 - 15 = 45`, so the distance between the stops is `45 - 20 = 25`. Plain inference latches onto the “15 miles” distractor; independent traces recover the two-step calculation and `25` wins the vote.

These outputs were generated on the public Hugging Face model with `mlx-lm 0.31.3`; they are not mocked transcripts. `siftsc demo` pins the questions, decoding seed, and demo routing threshold, then exits with an error if either behavior no longer reproduces.

When voting hurts, SiftSC stops after the correct first answer:

```text
Question    Darrell and Allen's ages are in the ratio 7:11. Their total
            age is 162. How old will Allen be in 10 years?

PLAIN       109  ✓
ALWAYS-SC   100  ✗   blind voting changed a correct answer
SIFTSC      109  ✓   voting skipped

[compute] actual=1 pass · Always-SC=5 passes · saved=4 (80.0%)
```

The demo finishes with the measured workload-level difference:

```text
MEASURED WORKLOAD · 400 PROMPTS
ALWAYS-SC   2,000 generation passes
SIFTSC        415 actual generation passes
SAVED       1,585 passes (79.2% less compute)
QUALITY     98.7% of Always-SC accuracy retained
```

<sub>Demo questions from the [GSM8K test set](https://github.com/openai/grade-school-math), released under the MIT License.</sub>

## Chat with the model

```bash
siftsc chat
```

SiftSC loads the same cached model and lets you choose the experience:

```text
Choose how the model should answer:
  1  plain   one deterministic answer
  2  siftsc  vote only when the router escalates
mode [2]>
```

Switch at any time without restarting the model:

```text
/plain     use ordinary one-pass inference
/siftsc    turn selective voting back on
/clear     clear the terminal
/exit      leave the session
```

You can also select a mode before launch:

```bash
siftsc chat --mode plain
siftsc chat --mode siftsc
```

Each chat turn is treated as an independent reasoning question because the bundled router was calibrated on math reasoning, not open-ended conversation history.

After every answer, the CLI reports both the current request and cumulative session usage:

```text
[siftsc] one pass · passes=1
[compute] actual=1 pass · Always-SC=5 passes · saved=4 (80.0%)
[session] actual=8 passes · Always-SC=15 passes · saved=7 (46.7%)
```

## The difference, in two numbers

| On Qwen2.5-0.5B MLX 4-bit | Always-SC@5 | SiftSC |
|---|---:|---:|
| Actual generation passes over 400 prompts | 2,000 | **415 · 79.2% fewer** |
| Accuracy relative to always-SC | 100% reference | **98.7% retained** |

The SiftSC pass count uses the CLI's conservative, auditable accounting: one routing draft on every prompt, plus five fresh voters on each of the three escalated prompts. The policy was measured on 400 pooled GSM8K and MATH-500 prompts. Its threshold was selected out-of-fold; it was not chosen on either demo question. Full confidence intervals, per-model results, checksums, and caveats remain available in [the benchmark report](docs/benchmarks/RESULTS.md), away from the quick-start path.

## How SiftSC works

```text
prompt → one deterministic draft → lightweight router
                                      ├─ accept the draft
                                      └─ sample 5 traces → parse → vote
```

The router reads statistics already produced during the first answer plus a few cheap prompt features. It does not call another language model. When the route stays greedy, the request ends after one model pass; when it escalates, the final answer comes only from five fresh sampled traces, matching the paper's SC@5 protocol.

Ask one question directly:

```bash
siftsc ask "If 3 notebooks cost £4 each, what is the total?"
siftsc ask "If 3 notebooks cost £4 each, what is the total?" --mode plain
```

Use any compatible local MLX directory instead of the default Hugging Face model:

```bash
siftsc chat --model /absolute/path/to/model
```

<details>
<summary><strong>Python API</strong></summary>

```python
from siftsc import MLXBackend, SiftSC, load_profile, math_prompt

question = "A shop sold 18 books on Monday and twice as many on Tuesday. Total?"
router = SiftSC(
    backend=MLXBackend("mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
    gate=load_profile("qwen05b-q4-confidence"),
    samples=5,
)
result = router(math_prompt(question), feature_text=question)

print(result.parsed_answer)
print(result.used_self_consistency, result.generation_passes)
print(result.vote_counts)
```

`SiftResult` preserves the route, every generation, parsed answers, vote counts, and the actual number of model passes for auditing.

</details>

<details>
<summary><strong>Research scope and honest limitations</strong></summary>

- The published evidence covers SC@5, Qwen-0.5B and Gemma-1B, FP16 and MLX 4-bit, two math benchmarks, and one generation seed per cell.
- The headline result uses the paper's exact Qwen MLX-Q4 checkpoint and prompt distribution. The downloadable community conversion is provided for immediate experience, not as a claim that the published threshold transfers perfectly to every conversion.
- The demo uses a documented, slightly more permissive routing threshold so both the paper checkpoint and public conversion reproduce the same correction. It demonstrates the mechanism; it is not an aggregate benchmark.
- Escalation performs one routing draft plus five voter samples. The paper's normalized operating-point cost compares the selected SC@5 route with always-SC@5; the CLI reports actual model passes.
- Recalibrate before changing the model, task distribution, prompt template, decoding settings, or risk tolerance. SiftSC is not a correctness verifier and should not be the sole decision-maker in high-stakes systems.

See [benchmark details](docs/benchmarks/RESULTS.md), [ethics review](docs/ETHICS_REVIEW.md), and [profile provenance](docs/benchmarks/profile_manifest.json).

</details>

## Develop

```bash
git clone https://github.com/heywanrong/SiftSC.git
cd SiftSC
python -m pip install -e '.[dev,mlx]'
ruff check . && mypy src/siftsc && pytest
```

The core router depends only on NumPy. MLX is loaded lazily, so custom backends can use the routing package on other platforms.

## Citation

```bibtex
@inproceedings{yang2026selfconsistencyhurts,
  title     = {When Self-Consistency Hurts: The Hidden Cost of Majority Voting in Small Language Models},
  author    = {Yang, Wanrong and Li, Lingfang and Zhang, Jun and Wojtczak, Dominik},
  booktitle = {International Conference on Neural Information Processing},
  year      = {2026},
  note      = {Forthcoming}
}
```

Apache-2.0. See [LICENSE](LICENSE).
