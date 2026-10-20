# Evaluation v1 - superseded (kept for transparency)

Superseded by evaluation v3 (D-025). Known problems:
- `sim_inputs_v1.yaml` says "UCF-Crime/CADP": CADP frames were never obtained; only UCF-Crime was used.
- "recall" counts any alarm in accident videos without checking timing (most have no annotated
  onset); the false-alarm rate pools road (pre-onset) and mostly non-road (UCF Normal) footage.
- The threshold was chosen with a false-alarm budget the validation data cannot resolve.
