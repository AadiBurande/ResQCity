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
