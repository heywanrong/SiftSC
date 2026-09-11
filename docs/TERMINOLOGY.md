# Terminology and metric lock

This file is the human-readable single source of truth for public wording.

| Category | Canonical term | Definition | Avoid |
|---|---|---|---|
| Project | SiftSC | The software package and repository | ThinkGate, TTC-Gate as the product name |
| Policy | selective self-consistency | Decide after the greedy pass whether to invoke SC | adaptive sampling when referring to invocation gating |
| Baseline | always-SC | Invoke SC for every prompt | standard SC when a precise baseline is needed |
| Gate action | invoke SC | Draw the remaining stochastic traces and vote | escalate to another model |
| Gate action | skip SC | Return the deterministic greedy answer | reject the prompt |
| Metric | skip rate | Fraction of prompts that do not invoke SC | savings rate without definition |
| Metric | cost ratio | Generation passes per prompt in units of a greedy pass | latency or energy savings |
| Metric | accuracy retention | Gated accuracy divided by always-SC accuracy in the same cell | accuracy guarantee |
| Event | SC helps | Greedy is wrong and SC is correct | improvement without direction |
| Event | SC hurts | Greedy is correct and SC is wrong | hallucination rate |

The paper's evidence is scoped to SC@5, two model families at or below 1B parameters, FP16/Q4, GSM8K/MATH-500, and one generation seed per cell. Public documentation must retain this scope.
