<div align="center">

<h1>
  <img src="docs/assets/logo.png" alt="Sifty, the winking SiftSC mascot" width="92" align="absmiddle">
  &nbsp;SiftSC
</h1>

### Think once. Vote only when it helps.

**Selective self-consistency for small, local language models.**<br>
No API key · No fine-tuning · No cloud inference

<a href="https://github.com/heywanrong/SiftSC/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/heywanrong/SiftSC/ci.yml?branch=main&style=flat-square&label=CI"></a>
<a href="https://www.python.org/"><img alt="Python 3.11 or newer" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
<a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache_2.0-7c3aed?style=flat-square"></a>

<br>

<img alt="Demo platform: Apple Silicon only" src="https://img.shields.io/badge/Demo-Apple_Silicon_only-111827?style=for-the-badge&logo=apple&logoColor=white">
<img alt="Bundled model: Qwen 0.5B 4-bit" src="https://img.shields.io/badge/Model-Qwen_0.5B_4--bit-0891b2?style=for-the-badge">
<img alt="Inference modes: Plain and SiftSC" src="https://img.shields.io/badge/Modes-Plain_%7C_SiftSC-7c3aed?style=for-the-badge">

<a href="#run-it-now"><img alt="Run the live demo" src="https://img.shields.io/badge/%E2%96%B6_RUN_THE_LIVE_DEMO-111827?style=for-the-badge"></a>
&nbsp;
<a href="#chat-with-the-model"><img alt="Start local chat" src="https://img.shields.io/badge/START_LOCAL_CHAT-0891b2?style=for-the-badge"></a>

</div>

> [!IMPORTANT]
> **The bundled demo and interactive chat currently require an Apple-Silicon Mac (M1 or newer).** Model inference uses MLX. The NumPy routing core is cross-platform through custom backends, but a built-in Linux or Windows model runner is not included yet.

Self-consistency can help a small model—but running it on every prompt spends five generation passes, and the majority can still overturn a correct answer. SiftSC makes that vote conditional: answer once, inspect cheap signals from the same pass, and sample five reasoning traces only when the router decides they are worth the compute.

**The result on our 400-prompt Qwen-0.5B workload: 79.2% fewer generation passes while retaining 98.7% of Always-SC accuracy.**

<table>
  <tr>
    <td width="33%" valign="top"><strong>⚡ Spend selectively</strong><br>Use one pass by default. Pay for five voters only when the router escalates.</td>
    <td width="33%" valign="top"><strong>🛡️ Protect good answers</strong><br>Avoid some cases where blind majority voting replaces a correct first answer.</td>
    <td width="33%" valign="top"><strong>🔎 See every decision</strong><br>Inspect the route, votes, actual passes, and compute saved after every answer.</td>
  </tr>
</table>

<p align="center">
  <img src="docs/assets/hero.svg" alt="SiftSC uses 79.2% fewer generation passes while retaining 98.7% of Always-SC accuracy">
</p>

<a id="run-it-now"></a>

## 🚀 One command. Two modes. Zero setup.

On an Apple-Silicon Mac with Python 3.11+, this single command installs SiftSC, downloads the tested public 4-bit model, and runs the complete comparison:

```bash
python -m pip install "siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git" && siftsc demo
```

<table>
  <tr>
    <td width="33%" valign="top"><strong>1 · 📦 Install</strong><br>The command installs the CLI and MLX backend.</td>
    <td width="33%" valign="top"><strong>2 · 🤖 Download</strong><br>The pinned 290 MB Qwen model is fetched once, then cached locally.</td>
    <td width="33%" valign="top"><strong>3 · 🧪 Compare</strong><br>Watch Plain, Always-SC, and SiftSC answer the same real questions.</td>
  </tr>
</table>

The bundled model is [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3). Its tested revision is pinned, so every later run starts from the same local checkpoint.

### 🧪 One model, two decisions: repair and protect

These are real outputs from the public model with `mlx-lm 0.31.3`, not mocked transcripts. The command pins the questions, decoding seed, and demo threshold—and fails loudly if the behavior no longer reproduces.

#### 🛠️ Repair: vote when one answer is shaky

Plain inference latches onto a distractor. SiftSC escalates, and independent traces recover the correct two-step calculation:

```text
Question  Henry made two stops during his 60-mile bike trip. He first
          stopped after 20 miles. His second stop was 15 miles before
          the end. How far did he travel between the two stops?

PLAIN    15  ✗   (1 deterministic pass)
SIFTSC   25  ✓   (votes: 20 · 15 · 25 · 25 · 30)
```

The second stop is at mile `60 - 15 = 45`; the distance between stops is `45 - 20 = 25`. Three of five independent traces reach `25`, so it wins the vote.

#### 🛡️ Protect: stop when voting would hurt

Here the first answer is already right. Always-SC votes itself into the wrong answer; SiftSC knows when to stop:

```text
Question    Darrell and Allen's ages are in the ratio 7:11. Their total
            age is 162. How old will Allen be in 10 years?

PLAIN       109  ✓
ALWAYS-SC   100  ✗   blind voting changed a correct answer
SIFTSC      109  ✓   voting skipped

[compute] actual=1 pass · Always-SC=5 passes · saved=4 (80.0%)
```

#### 📉 See the compute drop

The demo ends with the measured workload-level result—not a theoretical estimate:

```text
MEASURED WORKLOAD · 400 PROMPTS
ALWAYS-SC   2,000 generation passes
SIFTSC        415 actual generation passes
SAVED       1,585 passes (79.2% less compute)
QUALITY     98.7% of Always-SC accuracy retained
```

<sub>Demo questions from the [GSM8K test set](https://github.com/openai/grade-school-math), released under the MIT License.</sub>

<a id="chat-with-the-model"></a>

## 💬 Chat with the model

No configuration file is required. Start the cached model and choose ordinary one-pass inference or SiftSC:

```bash
siftsc chat
```

```text
Choose how the model should answer:
  1  plain   one deterministic answer
  2  siftsc  vote only when the router escalates
mode [2]>
```

### 🔀 Switch modes without reloading

Change the inference policy at any time while the model stays in memory:

```text
/plain     use ordinary one-pass inference
/siftsc    turn selective voting back on
/clear     clear the terminal
/exit      leave the session
```

Or select the mode before launch:

```bash
siftsc chat --mode plain
siftsc chat --mode siftsc
```

Each turn is treated as an independent reasoning question because the bundled router was calibrated on math reasoning, not open-ended conversation history.

### 📟 Watch compute savings live

After every answer, the CLI reports both the current request and cumulative session savings against Always-SC@5:

```text
[siftsc] one pass · passes=1
[compute] actual=1 pass · Always-SC=5 passes · saved=4 (80.0%)
[session] actual=8 passes · Always-SC=15 passes · saved=7 (46.7%)
```

## ⚡ The difference, in two numbers

| On Qwen2.5-0.5B MLX 4-bit | Always-SC@5 | SiftSC |
|---|---:|---:|
| Actual generation passes over 400 prompts | 2,000 | **415 · 79.2% fewer** |
| Accuracy relative to always-SC | 100% reference | **98.7% retained** |

The pass count uses conservative, auditable accounting: one routing draft for every prompt, plus five fresh voters for each of the three escalated prompts. The policy was measured on 400 pooled GSM8K and MATH-500 prompts; its threshold was selected out-of-fold and was not chosen on either demo question. Full confidence intervals, per-model results, checksums, and caveats live in [the benchmark report](docs/benchmarks/RESULTS.md), away from the quick-start path.

## 🧭 How SiftSC works

<table>
  <tr>
    <td width="33%" valign="top"><strong>① ✍️ Draft once</strong><br>Generate one deterministic answer and reuse signals already produced in that pass.</td>
    <td width="33%" valign="top"><strong>② 🧭 Route cheaply</strong><br>A tiny router adds a few prompt features. It never calls another language model.</td>
    <td width="33%" valign="top"><strong>③ 🗳️ Vote only if needed</strong><br>Easy prompts stop. Escalated prompts use five fresh traces matching SC@5.</td>
  </tr>
</table>

```text
prompt ──→ one draft ──→ route ──┬──→ accept now      · 1 pass
                                 └──→ sample + vote   · 6 actual passes
```

### ⌨️ Ask one question

Skip the interactive session and compare either policy directly:

```bash
siftsc ask "If 3 notebooks cost £4 each, what is the total?"
siftsc ask "If 3 notebooks cost £4 each, what is the total?" --mode plain
```

### 🧱 Bring your own local model

Point the CLI at any compatible local MLX directory instead of the default Hugging Face model:

```bash
siftsc chat --model /absolute/path/to/model
```

<details>
<summary><strong>🐍 Python API</strong></summary>

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
<summary><strong>🔬 Research scope and honest limitations</strong></summary>

- The published evidence covers SC@5, Qwen-0.5B and Gemma-1B, FP16 and MLX 4-bit, two math benchmarks, and one generation seed per cell.
- The headline result uses the paper's exact Qwen MLX-Q4 checkpoint and prompt distribution. The downloadable community conversion is provided for immediate experience, not as a claim that the published threshold transfers perfectly to every conversion.
- The demo uses a documented, slightly more permissive routing threshold so both the paper checkpoint and public conversion reproduce the same correction. It demonstrates the mechanism; it is not an aggregate benchmark.
- Escalation performs one routing draft plus five voter samples. The paper's normalized operating-point cost compares the selected SC@5 route with always-SC@5; the CLI reports actual model passes.
- Recalibrate before changing the model, task distribution, prompt template, decoding settings, or risk tolerance. SiftSC is not a correctness verifier and should not be the sole decision-maker in high-stakes systems.

See [benchmark details](docs/benchmarks/RESULTS.md), [ethics review](docs/ETHICS_REVIEW.md), and [profile provenance](docs/benchmarks/profile_manifest.json).

</details>

## 🧰 Develop

```bash
git clone https://github.com/heywanrong/SiftSC.git
cd SiftSC
python -m pip install -e '.[dev,mlx]'
ruff check . && mypy src/siftsc && pytest
```

The core router depends only on NumPy. MLX is loaded lazily, so custom backends can use the routing package on other platforms.

## 📚 Citation

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
