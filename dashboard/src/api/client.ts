import {
  Camera,
  DispatchDecision,
  Explanation,
  Hospital,
  Incident,
  Station,
  Unit,
} from "../types/schemas";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export interface KpiSummary {
  active_incidents: number;
  total_units: number;
  busy_units: number;
  fleet_utilisation: number;
  decisions: {
    proposed: number;
    approved: number;
    overridden: number;
    auto: number;
    override_rate: number;
  };
  mode: string;
  scenario: string;
}

export interface ScenarioStatus {
  running: boolean;
  scenario: string;
  mode: string;
  rain_mm_h: number;
}

export interface WardFeature {
  ward_id: string;
  name: string;
}

export const api = {
  async getHealth(): Promise<{ status: string; uptime_s: number }> {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async getWards(): Promise<WardFeature[]> {
    const res = await fetch(`${BASE_URL}/wards`);
    if (!res.ok) return [];
    return res.json();
  },

  async getCameras(): Promise<Camera[]> {
    const res = await fetch(`${BASE_URL}/cameras`);
    if (!res.ok) throw new Error(`Failed to fetch cameras: ${res.statusText}`);
    return res.json();
  },

  async getStations(): Promise<Station[]> {
    const res = await fetch(`${BASE_URL}/stations`);
    if (!res.ok) throw new Error(`Failed to fetch stations: ${res.statusText}`);
    return res.json();
  },

  async getHospitals(): Promise<Hospital[]> {
    const res = await fetch(`${BASE_URL}/hospitals`);
    if (!res.ok) throw new Error(`Failed to fetch hospitals: ${res.statusText}`);
    return res.json();
  },

  async getUnits(): Promise<Unit[]> {
    const res = await fetch(`${BASE_URL}/units`);
    if (!res.ok) throw new Error(`Failed to fetch units: ${res.statusText}`);
    return res.json();
  },

  async getIncidents(): Promise<Incident[]> {
    const res = await fetch(`${BASE_URL}/incidents`);
    if (!res.ok) throw new Error(`Failed to fetch incidents: ${res.statusText}`);
    return res.json();
  },

  async getDecisions(): Promise<DispatchDecision[]> {
    const res = await fetch(`${BASE_URL}/decisions`);
    if (!res.ok) throw new Error(`Failed to fetch decisions: ${res.statusText}`);
    return res.json();
  },

  async getDecision(id: string): Promise<{ decision: DispatchDecision; explanation: Explanation | null }> {
    const res = await fetch(`${BASE_URL}/decisions/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch decision ${id}: ${res.statusText}`);
    return res.json();
  },

  async triggerIncident(params: {
    camera_id?: string;
    lat?: number;
    lon?: number;
    severity?: number;
    incident_type?: string;
  }): Promise<Incident> {
    const res = await fetch(`${BASE_URL}/incidents/trigger`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Failed to trigger incident: ${res.statusText}`);
    return res.json();
  },

  async approveDecision(id: string, operatorId = "operator_pune"): Promise<DispatchDecision> {
    const res = await fetch(`${BASE_URL}/decisions/${id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operator_id: operatorId }),
    });
    if (!res.ok) throw new Error(`Failed to approve decision: ${res.statusText}`);
    return res.json();
  },

  async overrideDecision(
    id: string,
    reason: string,
    assignments: Array<{ unit_id: string; eta_s: number; role: string; hospital_id?: string | null }>,
    operatorId = "operator_pune"
  ): Promise<DispatchDecision> {
    const res = await fetch(`${BASE_URL}/decisions/${id}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        operator_id: operatorId,
        reason,
        assignments,
      }),
    });
    if (!res.ok) throw new Error(`Failed to override decision: ${res.statusText}`);
    return res.json();
  },

  async getKpis(): Promise<KpiSummary> {
    const res = await fetch(`${BASE_URL}/kpis`);
    if (!res.ok) throw new Error(`Failed to fetch KPIs: ${res.statusText}`);
    return res.json();
  },

  async getScenarioStatus(): Promise<ScenarioStatus> {
    const res = await fetch(`${BASE_URL}/scenario/status`);
    if (!res.ok) throw new Error(`Failed to fetch scenario status: ${res.statusText}`);
    return res.json();
  },

  async startScenario(scenario: string, mode = "assist"): Promise<ScenarioStatus> {
    const res = await fetch(`${BASE_URL}/scenario/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario, mode }),
    });
    if (!res.ok) throw new Error(`Failed to start scenario: ${res.statusText}`);
    return res.json();
  },

  async stopScenario(): Promise<ScenarioStatus> {
    const res = await fetch(`${BASE_URL}/scenario/stop`, {
      method: "POST",
    });
    if (!res.ok) throw new Error(`Failed to stop scenario: ${res.statusText}`);
    return res.json();
  },
};
