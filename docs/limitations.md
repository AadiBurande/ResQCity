# Limitations and threats to validity (living document)

Each entry names the limitation, which results it can affect, and what mitigates it. Entries are
added as phases progress; the final write-up (phase 11) draws on this list.

## Data and geography (phase 2)

| # | Limitation | Affects | Mitigation |
|---|---|---|---|
| L1 | Ward boundaries are hand-traced from PMC PDF maps (DataMeet); positional error of tens of metres is likely. | Ward assignment of nodes near boundaries; ward-level equity | Errors are small relative to ward sizes (median 4.6 km²); nodes are assigned by point-in-polygon with nearest-ward fallback. |
| L2 | Travel speeds are assumptions hand-calibrated to one reported rush-hour figure (D-014). Off-peak times are probably too fast. | All response-time results | Global speed scale is a phase-10 sensitivity factor; hold-out route check is an open action. |
| L3 | Spatial demand follows OSM road-class density, not observed crash locations. | Ward-level demand and **equity results**, which partly reflect how densely OSM is mapped | Alternative weighting (e.g. uniform per km of arterial road) as a phase-10 sensitivity run; stated in every equity claim. |
| L4 | Incident rates, type mix, injury and fire probabilities are synthetic assumptions (all UNVERIFIED). | Share of multi-agency incidents, hence the size of the RQ2 effect | All are `SourcedValue`; sensitivity sweeps in phase 10; replace with cited figures where possible. |
| L5 | Flooding uses a river-distance buffer only. Nalas (storm-water streams), low spots and drainage failures are not modelled; stage fractions are illustrative. | Monsoon scenario results | Flood stages are UNVERIFIED parameters; the monsoon scenario is presented as a stress test, not a forecast. |
| L6 | Ganeshotsav routes and timings are approximate (street names, 2.5 km radius around Mandai, assumed dates). | Ganeshotsav scenario results | Route match counts are recorded; dates and routes flagged for checking against Traffic Police notices. |
| L7 | OSM facility data are incomplete or noisy: 14 fire stations, 591 `amenity=hospital` features including small nursing homes. Ambulance bases are assumed. | Fleet locations, nearest-unit times | Ambulance bases and hospital capabilities are labelled synthetic; fleet configuration (phase 6/7) selects bases explicitly. |
| L8 | Static-bucket routing: each trip uses edge times at departure (no change of speed during the trip). | Long trips spanning a period boundary | Exact time-dependent routing in phase 6.1. |
| L9 | The same rule table defines ground-truth requirements and the estimates derived from video and calls. | Could flatter CCTV-based policies | An explicit requirement-estimation error model, applied to calls and CCTV alike (phases 4 and 7). |

## Vision (phase 3)

| # | Limitation | Affects | Mitigation |
|---|---|---|---|
| L10 | Accident footage is UCF-Crime (mixed countries, 320×240 CCTV), not Pune cameras: domain shift in viewpoint, traffic mix (two/three-wheelers), resolution and weather. | Detector recall, delay and false-alarm rates used by the simulator | Report as measured-on-public-data; detector parameters are sensitivity factors in phase 10; re-evaluate if Pune footage becomes available. |
| L11 | Only 23 accident videos have annotated onsets (12 in test): detection delay and video-level recall have wide uncertainty. | RQ1 (detection earlier than calls) | Report intervals/raw delays, not only point estimates; RQ1 is a curve over call-delay assumptions anyway (master plan §6.3). |
| L12 | Plate blurring uses the lower band of each vehicle box (no plate detector); face blurring depends on detector recall at low resolution; people the detector misses are not blurred. | Privacy guarantee of the preview stream | Whole person boxes are additionally blurred; regions grown to ≥ 24 px; privacy check reports per-region detail reduction (D-023); only event metadata leaves `vision/`. |
| L15 | The accident detector localises poorly in time on UCF-Crime (within-video AUROC ≈ 0.6; 1/12 test onsets detected at the val operating point; v2 operating point chosen after seeing v1 test results, D-022). | RQ1 and every CCTV-triggered policy in the simulator | Detector behaviour enters the simulator only through measured `sim_inputs.yaml` with intervals; phase 10 reports results as a function of detector quality (break-even analysis). |
| L13 | Fire/smoke detector is trained on D-Fire (mostly wildfire/building scenes), not vehicle fires on Indian roads. | Fire event recall | Fire events carry the detector's (uncalibrated) box confidence and need persistence over several frames; treated as a separate, uncalibrated cue. |
| L14 | COCO classes have no auto-rickshaw; three-wheelers are detected as car/motorcycle, which can shift severity cues (vehicle classes). | Required-services estimate | Severity uses class only for heavy vehicles (truck/bus); fine-tuning on Indian traffic data (e.g. IDD) is future work. |
