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
<img alt="Scope: checkable answers only" src="https://img.shields.io/badge/Scope-Checkable_answers_only-7c3aed?style=for-the-badge">

</div>

> [!IMPORTANT]
> **Two limits to know before you install.**
>
> 1. **Apple Silicon only.** The bundled demo and chat currently require an Apple-Silicon Mac (M1 or newer); inference uses MLX. The NumPy routing core runs anywhere through a custom backend, but a built-in Linux or Windows model runner is not included yet.
> 2. **Only questions with a checkable answer.** SiftSC votes on answers it can count, such as the number at the end of a word problem. Its router and thresholds were calibrated on math benchmarks. Any other question is answered once by the plain model, without voting, and the CLI says so.

SiftSC answers once, reads cheap signals from that pass, and samples five more reasoning traces only when a small router says the vote is worth it. On a 400-prompt Qwen-0.5B workload it used **79.2% fewer generation passes while retaining 98.7% of always-SC accuracy**.

```text
prompt ──→ one draft ──→ route ──┬──→ accept now      · 1 pass
                                 └──→ sample + vote   · 6 actual passes
```

<p align="center">
  <img src="docs/assets/hero.svg" alt="SiftSC uses 79.2% fewer generation passes while retaining 98.7% of Always-SC accuracy">
</p>

## 🚀 Run it

One line. It installs [uv](https://docs.astral.sh/uv/) if needed (a tool manager that also fetches Python itself, so no Python setup is required), then SiftSC with the model runner and the full-screen interface:

```bash
curl -fsSL https://raw.githubusercontent.com/heywanrong/SiftSC/main/install.sh | sh
```

Then just type:

```bash
siftsc
```

`siftsc` alone opens the full-screen interface; `siftsc demo` plays two verified cases first and then hands you the prompt. The first launch downloads the pinned 290 MB [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3) once. Re-run the same line to upgrade; `siftsc --version` shows which release you have.

<details>
<summary><strong>📦 Other ways to install</strong></summary>

The installer needs the repository to be public. While it is a private preview, GitHub must be reachable and your Git client authenticated as a collaborator; then either line works:

```bash
uv tool install --python 3.12 "siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git"
```

```bash
python -m pip install -q --upgrade "siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git"
```

After the PyPI release the spec becomes simply `"siftsc[mlx]"`. The `mlx` extra includes the interface; `siftsc[ui]` adds only the interface, and the bare package keeps the NumPy router for custom backends.

</details>

<details>
<summary><strong>🧯 Install problems</strong></summary>

`Failed to connect to github.com port 443` is raised by `git clone` before SiftSC starts. Check connectivity and private-repository access:

```bash
curl -I https://github.com
gh auth status
gh auth setup-git
git ls-remote https://github.com/heywanrong/SiftSC.git HEAD
```

From a local clone, skip GitHub entirely:

```bash
cd /path/to/SiftSC
python -m pip install -q --upgrade '.[mlx]' && siftsc demo
```

Library output during loading is hidden; `SIFTSC_VERBOSE=1` shows it and `SIFTSC_NO_ANIMATION=1` disables the spinner. `siftsc demo --no-chat` exits after the showcase for scripts and CI.

</details>

## 🧪 What the demo shows

Real outputs from the public model with `mlx-lm 0.31.3`. The command pins the questions, seed and threshold, and fails loudly if the behaviour stops reproducing.

**Repair.** Plain inference latches onto a distractor; the router calls a vote and three of five traces recover `60 − 15 − 20 = 25`.

```text
🗳️  CASE 1 · VOTE WHEN IT HELPS
Question   Henry made two stops during his 60-mile bike trip. He first stopped
           after 20 miles. His second stop was 15 miles before the end of the
           trip. How many miles did he travel between his first and second
           stops?
Expected   25
PLAIN      15  ✗   1 pass
SIFTSC     25  ✓   6 passes · votes 20 · 15 · 25 · 25 · 30
```

**Protect.** The first answer is already right and blind voting would overturn it, so the router skips the vote.

```text
🛡️  CASE 2 · SKIP WHEN VOTING HURTS
Question   Darrell and Allen's ages are in the ratio of 7:11. If their total age
           now is 162, calculate Allen's age 10 years from now.
Expected   109
PLAIN      109  ✓   1 pass
ALWAYS-SC  99  ✗   5 passes · blind voting changed a right answer
SIFTSC     109  ✓   1 pass · vote skipped
   ⚡ plain      █░░░░░   1 pass
   👥 always-SC  █████░   5 passes
   🧭 siftsc     █░░░░░   1 pass   ✅ saved 4 (80.0%)
```

**Measured workload.** One chart cell is one generation pass.

```text
📉 MEASURED WORKLOAD · 400 PROMPTS
   👥 always-SC  ████████████████████████████████  2,000 passes
   🧭 siftsc     ███████░░░░░░░░░░░░░░░░░░░░░░░░░    415 passes
   ✅ saved 1,585 passes · 79.2% less compute
   🎯 98.7% of always-SC accuracy retained
```

<sub>Demo questions from the [GSM8K test set](https://github.com/openai/grade-school-math), MIT License. The losing always-SC answer can differ between machines; what is checked is that the blind vote loses `109` and SiftSC keeps it.</sub>

## 🖥️ Full-screen interface

`siftsc` (or `siftsc ui`) opens Sifty's full-screen chat: the conversation on the left, the cost of the current question and the session on the right, a spinner while the model works, and a footer with the keys. The same slash commands work there, plus `Ctrl+T` to cycle the mode, `Ctrl+G` for the session chart, `Ctrl+L` to clear and `Ctrl+Q` to quit.

```text
┌ SiftSC · 🧭 siftsc · votes only when the router escalates ──────────────────────────┐
│ 💬 you · siftsc                                       │ 📈 This question             │
│    Henry made two stops during his 60-mile bike trip… │ ⚡ plain    █░░░░░  1        │
│ 🤖 Sifty                                              │ 👥 always   █████░  5        │
│    … 60 - 35 = 25 miles between the stops. Answer: 25 │ 🧭 siftsc   ██████  6        │
│ 🎯 Answer: 25 · 🗳️ vote called · 6 passes · 2.8s      │                              │
│                                                        │ 📊 Session                   │
│                                                        │ 2 questions · 1 vote         │
│                                                        │ spent 7 · always-SC 10       │
│ > Ask a question · Enter to send · /help for commands  │ ✅ saved 3 (30.0%)           │
│ ^T Mode  ^G Session  ^L Clear  ^Q Quit                                                │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## 💬 Line-mode chat

Prefer plain scrolling output? After the demo, type at the `💬 you` prompt, or start fresh with `siftsc chat`. Every answer ends with the route the router took and a compute chart:

```text
💬 you · siftsc > Henry made two stops during his 60-mile bike trip...
✨ Answer ready · 2.9s

🤖 Sifty
   ... he traveled 60 - 35 = 25 miles between the first and second stops. Answer: 25.

🎯 Answer: 25 · 🗳️ vote called · votes 20 · 15 · 25 · 25 · 30 · 6 passes · 2.9s
   ⚡ plain      █░░░░░   1 pass   (the draft said 15)
   👥 always-SC  █████░   5 passes
   🧭 siftsc     ██████   6 passes 🛠️ vote changed the draft · ⚠️ extra 1 (+20.0%)
📊 session · 2 questions · 7 vs 10 passes · saved 3 (30.0%) vs always-SC
```

| Command | What it does |
|---|---|
| `/siftsc` | vote only when the router escalates (default) |
| `/plain` | one deterministic pass, never votes |
| `/compare` | run plain, always-SC and siftsc on the same question; shows each answer, its passes and seconds |
| `/stats` | draw the session compute chart |
| `/math <q>` · `/talk <q>` | force the reasoning route, or a one-pass chat answer, for one question |
| `/help` · `/clear` · `/exit` | the usual |

Questions without a checkable answer are answered once, in the language of the question, and never voted:

```text
💬 you · siftsc > 中国的首都在哪

🤖 Sifty
   中国的首都是北京。

💬 General question · answered once in chat style · 0.1s
   Votes are for reasoning questions with a checkable answer · try /math <q>
```

One question from the shell: `siftsc ask "..."` with `--mode plain|siftsc|compare`, `--question-type math|general`, `--json`, or `--model /path/to/mlx/model`. Arrow keys and history work at the prompt; Ctrl-C stops the current answer.

## 📏 Scope

- **Checkable answers only.** Voting counts parsed answers such as numbers or short strings. Open-ended text has no majority to count, so SiftSC does not apply to it.
- **The evidence is math.** SC@5 on GSM8K and MATH-500 with Qwen-0.5B and Gemma-1B, FP16 and MLX 4-bit, one generation seed per cell. The demo and public-model chat use a documented, slightly more permissive threshold (`0.84`) so the public checkpoint reproduces the correction; they show the mechanism, not the aggregate benchmark.
- **Thresholds are distribution-specific.** Recalibrate before changing the model, quantization, task, prompt template or decoding. `siftsc.calibration.fit_logistic` is the starting point for a new profile.
- **Not a correctness verifier.** A skipped vote means the router expected no gain, not that the answer is right. Do not make it the sole decision-maker in high-stakes settings.
- **Honest accounting.** Passes are the transparent compute proxy: one routing draft per prompt plus five fresh voters per escalation, so an escalated request costs six passes. Seconds and tokens are measured on your machine.

Details: [benchmark report](docs/benchmarks/RESULTS.md) · [ethics review](docs/ETHICS_REVIEW.md) · [profile provenance](docs/benchmarks/profile_manifest.json) · [terminology](docs/TERMINOLOGY.md).

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

`SiftResult` keeps the route, every generation, parsed answers, vote counts and the actual number of passes. The core depends only on NumPy; MLX loads lazily, so a custom backend can use the router on other platforms.

</details>

## 🧰 Develop

```bash
git clone https://github.com/heywanrong/SiftSC.git && cd SiftSC
python -m pip install -e '.[dev,mlx]'
ruff check . && mypy src/siftsc && pytest
```

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
