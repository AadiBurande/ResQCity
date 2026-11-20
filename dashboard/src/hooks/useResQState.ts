import { useCallback, useEffect, useState } from "react";
import { api, KpiSummary, ScenarioStatus } from "../api/client";
import { wsClient, WebSocketEvent } from "../api/websocket";
import {
  Camera,
  DispatchDecision,
  Explanation,
  Hospital,
  Incident,
  Station,
  Unit,
} from "../types/schemas";

export function useResQState() {
  const [cameras, setCameras] = useState<Record<string, Camera>>({});
  const [stations, setStations] = useState<Record<string, Station>>({});
  const [hospitals, setHospitals] = useState<Record<string, Hospital>>({});
  const [units, setUnits] = useState<Record<string, Unit>>({});
  const [incidents, setIncidents] = useState<Record<string, Incident>>({});
  const [decisions, setDecisions] = useState<Record<string, DispatchDecision>>({});
  const [explanations, setExplanations] = useState<Record<string, Explanation>>({});
  const [kpis, setKpis] = useState<KpiSummary | null>(null);
  const [scenarioStatus, setScenarioStatus] = useState<ScenarioStatus>({
    running: false,
    scenario: "normal",
    mode: "assist",
    rain_mm_h: 0,
  });

  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastNotification, setLastNotification] = useState<string | null>(null);

  // Initial load
  const refreshStatic = useCallback(async () => {
    try {
      const [cams, stns, hosps, unts, incs, decs, kp, sc] = await Promise.all([
        api.getCameras().catch(() => []),
        api.getStations().catch(() => []),
        api.getHospitals().catch(() => []),
        api.getUnits().catch(() => []),
        api.getIncidents().catch(() => []),
        api.getDecisions().catch(() => []),
        api.getKpis().catch(() => null),
        api.getScenarioStatus().catch(() => ({
          running: false,
          scenario: "normal",
          mode: "assist",
          rain_mm_h: 0,
        })),
      ]);

      setCameras(Object.fromEntries(cams.map((c) => [c.id, c])));
      setStations(Object.fromEntries(stns.map((s) => [s.id, s])));
      setHospitals(Object.fromEntries(hosps.map((h) => [h.id, h])));
      setUnits(Object.fromEntries(unts.map((u) => [u.id, u])));
      setIncidents(Object.fromEntries(incs.map((i) => [i.id, i])));
      setDecisions(Object.fromEntries(decs.map((d) => [d.id, d])));
      setKpis(kp);
      setScenarioStatus(sc);
      setConnected(true);

      // Fetch explanations for decisions
      for (const d of decs) {
        api.getDecision(d.id).then((res) => {
          if (res.explanation) {
            setExplanations((prev) => ({ ...prev, [d.id]: res.explanation! }));
          }
        }).catch(() => {});
      }
    } catch (e) {
      console.warn("Error refreshing static state:", e);
    }
  }, []);

  useEffect(() => {
    refreshStatic();
    wsClient.connect();

    const unsubscribe = wsClient.subscribe((event: WebSocketEvent) => {
      setConnected(true);
      switch (event.type) {
        case "init": {
          const initData = event.data;
          if (initData.cameras) setCameras(initData.cameras);
          if (initData.stations) setStations(initData.stations);
          if (initData.hospitals) setHospitals(initData.hospitals);
          if (initData.units) setUnits(initData.units);
          if (initData.incidents) setIncidents(initData.incidents);
          if (initData.decisions) setDecisions(initData.decisions);
          if (initData.kpis) setKpis(initData.kpis);
          break;
        }
        case "units_tick": {
          if (event.data?.units && Array.isArray(event.data.units)) {
            setUnits((prev) => {
              const next = { ...prev };
              for (const u of event.data.units) {
                next[u.id] = u;
              }
              return next;
            });
          }
          break;
        }
        case "incident_update": {
          const inc: Incident = event.data;
          setIncidents((prev) => ({ ...prev, [inc.id]: inc }));
          setSelectedIncidentId(inc.id);
          setLastNotification(`New incident detected: ${inc.id} (${inc.ward_id})`);
          break;
        }
        case "decision_ready": {
          const dec: DispatchDecision = event.data;
          setDecisions((prev) => ({ ...prev, [dec.id]: dec }));
          api.getDecision(dec.id).then((res) => {
            if (res.explanation) {
              setExplanations((prev) => ({ ...prev, [dec.id]: res.explanation! }));
            }
          }).catch(() => {});
          setLastNotification(`Dispatch decision ready for ${dec.incident_id}`);
          break;
        }
        case "decision_update": {
          const dec: DispatchDecision = event.data;
          setDecisions((prev) => ({ ...prev, [dec.id]: dec }));
          break;
        }
        case "scenario_status":
        case "scenario_state": {
          setScenarioStatus((prev) => ({ ...prev, ...event.data }));
          setLastNotification(
            `Scenario updated: ${event.data.scenario?.toUpperCase()} (${event.data.mode?.toUpperCase()})`
          );
          break;
        }
        case "hospital_prealert": {
          setLastNotification(`Hospital pre-alert: ${event.data.hospital_id} for ${event.data.incident_id}`);
          break;
        }
        case "green_corridor_active": {
          setLastNotification(`Green corridor active for incident ${event.data.incident_id}`);
          break;
        }
      }
    });

    // KPI periodic poll every 5s
    const kpiTimer = setInterval(() => {
      api.getKpis().then(setKpis).catch(() => {});
    }, 5000);

    return () => {
      unsubscribe();
      clearInterval(kpiTimer);
    };
  }, [refreshStatic]);

  return {
    cameras,
    stations,
    hospitals,
    units,
    incidents,
    decisions,
    explanations,
    kpis,
    scenarioStatus,
    selectedIncidentId,
    setSelectedIncidentId,
    selectedCameraId,
    setSelectedCameraId,
    connected,
    lastNotification,
    refreshStatic,
  };
}
