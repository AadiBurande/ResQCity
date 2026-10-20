# Evaluation v2 - superseded (kept for transparency)

Superseded by evaluation v3 (D-025). Known problems (research audit, 2026-10-20):
- "video_recall" (0.35) counts any alarm in accident videos without checking timing; timing-checked
  detection was 1/12 onsets.
- `false_alarms_per_camera_hour` (7.2/h) pools road footage before onsets (all 4 false alarms, in
  0.032 h) with mostly non-road UCF Normal footage (0 false alarms in 0.53 h).
- "Well calibrated" (ECE 0.013) is driven by easy normal windows; calibration within onset videos
  and above the threshold is much worse (see v3 report.json).
- v2 was adopted after the v1 test result had been seen.
