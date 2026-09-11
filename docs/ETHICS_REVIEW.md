# Research software release ethics and compliance review

- Review object: SiftSC v0.1.0 private GitHub preview
- Type: research software and aggregate benchmark artifacts
- Reviewer / date: Codex-assisted maintainer review / 2026-09-11
- Applicable context: United Kingdom / international open-source distribution

## Risk table

| Status | Risk category | Evidence and location | Compliance action | Third-party review? |
|---|---|---|---|---|
| NOTE | Academic integrity / attribution | Method and quantitative claims are attributed to the forthcoming paper in `README.md` and `CITATION.cff`. | Keep citation metadata synchronized when DOI/proceedings details become available. | No for private preview; publisher metadata later |
| PASS | Data fabrication or falsification | Tables and plots are regenerated from 1,600 cached prompt records; exact source SHA-256 hashes are in `docs/benchmarks/*.csv` and profile JSON. | Release script remains deterministic and is tested against the original local tree. | No |
| PASS | Image manipulation | Both PNG charts are direct aggregate plots; architecture SVG is explicitly a conceptual workflow graphic. No photographic or generative scientific image is used. | Keep plotting script and aggregate CSVs in the repository. | No |
| PASS | Citation and retraction status | The software README cites only the authors' forthcoming paper and official project links; it makes no literature-derived claim requiring a reference audit. | Add DOI and final proceedings URL only after publisher confirmation. | No |
| NOTE | Authorship | Software metadata names Wanrong Yang as package author; preferred paper citation lists the four paper authors. | Confirm software contributor credits before a public release. | Maintainer/co-authors before public release |
| PASS | Privacy / human participants | No human-subject data or PII is published. One public GSM8K prompt and its generated demo outputs are included; the local smoke report scrubs filesystem paths. | Continue secret/absolute-path scans before every release. | No IRB/ethics committee required for this software artifact |
| PASS | Animal research | Not applicable. | None. | No IACUC required |
| NOTE | Copyright and licenses | New source is Apache-2.0. One GSM8K test example is attributed to its MIT-licensed upstream repository. Model weights, paper PDFs, and third-party templates are excluded. Profiles contain only derived numeric coefficients and provenance hashes. | Preserve the GSM8K attribution and confirm institutional/IP expectations before making the repository public. | Institutional/legal review only if required by employment terms |
| PASS | Overclaiming | README includes the one-seed, task, model-size, precision, retention-CI, and threshold-selection limitations. | Preserve `docs/TERMINOLOGY.md` as wording lock. | No |
| PASS | Deceptive ghostwriting | Not applicable to a software implementation. | Maintainer reviews and owns all public claims and code. | No |
| NOTE | Patent / ownership | No patent claim is made. Potential institutional ownership has not been adjudicated in this code review. | Check University/employer policy before public commercialization or patent filing. | Institutional/legal only if applicable |
| PASS | Software authenticity | Package builds, unit tests run, and both one-pass and five-voter paths execute against local and public Qwen MLX-Q4 models. | Preserve the scrubbed smoke report and CI. | No |

## Contribution and AI-use note

- Paper authorship and CRediT roles are outside the scope of this software packaging pass and are not altered here.
- Wanrong Yang is listed as the initial software package author; additional software contributors should be credited from Git history.
- AI-assisted engineering was used to structure, implement, test, and document the private-preview repository. The maintainer remains responsible for technical accuracy, licensing, research claims, and the decision to publish.

## Conclusion

- Overall: pass for a private preview, with warnings to confirm contributor credit, final citation metadata, and institutional IP expectations before a public release.
- Blocking items for the requested private push: none found.
- Third-party review required now: none. Before public/commercial release: author and institutional review as applicable.
- Residual uncertainty: the final DOI/proceedings record is not yet available; bundled thresholds remain research estimates rather than guarantees.
