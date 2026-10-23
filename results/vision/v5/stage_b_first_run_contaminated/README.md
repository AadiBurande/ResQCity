Superseded stage-B run (all candidates loaded one after another in ONE process), run on
2026-10-23 between 16:52 IST (fine-tuned verifier finished, `../verifier_ft/*.meta.json`) and
17:42 IST (`aa92b35`, one process per candidate; clean results `6a2aa14`). Kept for the
record (D-027); not used for any decision. No provenance sidecar was written for it.

It is NOT the run that triggered amendment 3. Timeline (from the working-session log; the
earlier outputs were overwritten, so only this record and the log remain):
- 11:46-12:07 IST: an earlier single-process stage-B attempt logged validation results for the
  zero-shot Qwen3.5-2B (best 44/69 detections kept) and Qwen3-VL-2B (best 48/69). These are the
  figures quoted in amendment 3.
- 12:11 IST: amendment 3 committed (`ffae3ee`). At that moment TAU-R1-Classifier, Molmo2-4B and
  Gemma-4-E2B had NOT been scored on validation. TAU-R1 had only a one-video feasibility smoke
  test on a UCF-Crime training video.
- 16:52 IST: the fine-tuned candidate finished training; this full single-process run then
  scored every candidate, including it.

Latencies of candidates loaded after Gemma 4 / Molmo2 were inflated, probably by GPU memory
left behind by the earlier models (not measured directly). The fine-tuned Qwen3-VL took 10.2 s
here, against 0.40 s alone in a fresh process (`../verifier_smoke/qwen3_vl_2b_ft.json`).
The clean re-run (one process per candidate) is `../stage_b.csv`.
