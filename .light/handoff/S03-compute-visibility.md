---
session_no: S03
suggested_title: "[SiftSC] S04 private preview review"
parent_session: S02
project: siftsc
date: 2026-09-11
---

## Current stage

The private preview now makes both sides of selective self-consistency visible: compute saved and harmful voting avoided.

## Completed

- Every `ask` and `chat` answer reports actual generation passes, the Always-SC@5 baseline, and percentage saved or added.
- Interactive chat also reports cumulative session savings.
- The public-model demo now reproduces two cases: voting repairs `15 -> 25`, and selective skipping protects `109` from an Always-SC result of `100`.
- The README headline now uses exactly two results: 79.2% fewer generation passes and 98.7% Always-SC accuracy retained.
- Deployment accounting is explicit: 2,000 Always-SC passes versus 415 actual SiftSC passes over 400 prompts.

## Accounting note

Generation passes are used as the transparent compute proxy. The 415 count includes one routing draft for all 400 prompts and five fresh voter generations for each of three escalations. It is not presented as an identical percentage reduction in wall-clock time, energy, or generated tokens.
