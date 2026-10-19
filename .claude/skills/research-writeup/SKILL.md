---
name: research-writeup
description: Write or update the methodology, results, discussion or limitations sections of the project report or paper from files in results/. Use when drafting any report, paper, viva notes or summary of findings.
---

# Writing up ResQCity results

1. Read `results/summary.md`, `results/stats/*.csv` and the experiment configs first.
2. Every number in the text must be inserted by `python -m paper.build` from results files.
   Write placeholders like `{{stats.B3_vs_B0.last_arrival.mean_diff}}`, never typed numbers.
3. Structure (IMRaD): Introduction, Related work, Methodology (research design, system,
   models, simulation, experiment design, statistical analysis), Results (one subsection per RQ),
   Discussion, Threats to validity, Limitations, Ethics, Conclusion, Future work.
4. For each RQ state the hypothesis, the test used, the result, effect size with CI, and a
   plain-English reading. Say clearly when a hypothesis is not supported.
5. Separate what is measured (vision metrics on public datasets) from what is simulated
   (dispatch outcomes on synthetic incidents) in every claim.
6. Do not invent references. Use `[CITE: topic]` placeholders for the student to fill.
7. Ask the `research-auditor` subagent to check the draft for unsupported claims before finishing.
