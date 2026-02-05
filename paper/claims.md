# Claim Map

- id: C1
  claim: Search grounding improves no-leak accuracy by roughly 10 points on the dev set.
  evidence: fig:grounding-conservative (paper/figures/grounding_impact_conservative.pdf)
  location: sections/04_results.tex
  last_verified_commit: unknown

- id: C2
  claim: Gemini 3 Flash performs better than Gemini 3 Pro on temporal grounding.
  evidence: fig:grounding-delta-search; fig:test-delta-search
  location: sections/05_discussion.tex
  last_verified_commit: unknown

- id: C3
  claim: Filtering pass processed 30,549 SFT samples, 29,510 preference samples, and three RLVR datasets.
  evidence: fig:sft, fig:dpo, fig:rlvr
  location: sections/06_data_filtering.tex
  last_verified_commit: unknown
