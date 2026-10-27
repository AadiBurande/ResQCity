# Routing and the green corridor

Status: phase 6.1 (D-031). Code: `src/resqcity/optimization/routing.py`. Parameters:
`configs/optimization.yaml` → `routing`. Tests: `tests/optimization/test_routing_fleet.py`.

## 1. Travel times

Edge travel times come from phase 2 and are **not** re-derived here:
$$
t_e(\tau) = \frac{\text{length}_e}{v_e \cdot f_{\text{class}(e)}(\text{hour}(\tau)) \cdot
f_{\text{rain}}(\tau) \cdot f_{\text{scenario}}(e, \tau)}
$$
with the class, rain, flood and festival factors of `geo.scenario.ScenarioState`. Within an hour
bucket the times are constant (the static-bucket approximation, L2), so a shortest path is an
ordinary Dijkstra rather than a time-dependent search.

$\eta_{u,i}$, the ETA of unit $u$ to incident $i$, is the shortest path from $u$'s current node
to $i$'s node at the current hour bucket, or $\infty$ if no path exists (flooded or disconnected).

## 2. Computing the ETA table the cheap way round

A dispatch batch has many units and few incidents ($|U| \approx 91$, $|I| \approx 10$). A forward
search from each unit costs $|U|$ Dijkstras; a search on the **transposed** graph from each
incident costs $|I|$ and yields the travel time from every node *to* that incident. The table is
therefore built with `RoadNetwork.shortest_times(..., reverse=True)` over the incidents and the
unit rows are read off. Both directions are verified to agree in
`test_eta_table_matches_a_direct_shortest_path`.

## 3. The green corridor

Police holding junctions ahead of an ambulance shorten its remaining journey by a factor
$\rho \in (0, 1]$ (`routing.green_corridor.rho`; $\rho = 1$ disables it and is the phase-10
ablation).

Let $\pi_i = \min_{p \in P} \eta_{p,i}$ be the earliest any police unit could reach $i$.

- Ambulance $u$ is **eligible** at $i$ when $\rho\,\eta_{u,i} \ge \pi_i$: some police unit could
  be in place by the time the sped-up ambulance arrives.
- For an eligible pair, $\tilde\eta_{u,i} = \rho\,\eta_{u,i}$; otherwise
  $\tilde\eta_{u,i} = \eta_{u,i}$.

Eligibility is computed here, before the solver runs, so the assistance is **data** and not a
decision variable. It is necessary but not sufficient: the dispatch model additionally requires a
police unit actually *sent to this incident* that is fast enough to realise the corridor — see
`docs/models/dispatch.md` §6 (C7). Treating eligibility alone as sufficient would credit a
corridor to a police unit dispatched elsewhere, which an OR review of this phase found and which
is now a regression test.

## 4. Assumptions

1. **One factor for the whole route.** Whether police can clear a route in Pune traffic, and by
   how much, is UNVERIFIED; $\rho = 0.85$ is assumed and swept in phase 10.
2. **No signal pre-emption model.** The corridor stands in for both police escorting and signal
   pre-emption; they are not distinguished.
3. **Static within the hour bucket** (L2), so a long journey crossing a bucket boundary uses the
   departure bucket throughout.
4. **Units travel from their current node.** En-route reassignment inside a batch is not modelled
   (`docs/models/dispatch.md` §8).
