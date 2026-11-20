import React, { useState } from "react";
import { AlertTriangle, CloudRain, Flame, Play, Power, ShieldAlert, Sparkles, StopCircle, Zap } from "lucide-react";
import { api, ScenarioStatus } from "../../api/client";
import { Camera, Unit } from "../../types/schemas";

interface ScenarioManagerProps {
  scenarioStatus: ScenarioStatus;
  cameras: Record<string, Camera>;
  units: Record<string, Unit>;
  onTriggerSuccess: () => void;
}

export const ScenarioManager: React.FC<ScenarioManagerProps> = ({
  scenarioStatus,
  cameras,
  units,
  onTriggerSuccess,
}) => {
  const [selectedScenario, setSelectedScenario] = useState(scenarioStatus.scenario || "normal");
  const [selectedMode, setSelectedMode] = useState(scenarioStatus.mode || "assist");
  const [isStarting, setIsStarting] = useState(false);

  // Manual Trigger State
  const [targetCamera, setTargetCamera] = useState<string>("");
  const [customLat, setCustomLat] = useState("18.5204");
  const [customLon, setCustomLon] = useState("73.8567");
  const [severity, setSeverity] = useState(0.85);
  const [incidentType, setIncidentType] = useState("accident");
  const [isTriggering, setIsTriggering] = useState(false);

  const handleStart = async () => {
    setIsStarting(true);
    try {
      await api.startScenario(selectedScenario, selectedMode);
    } catch (e) {
      console.error("Failed to start scenario:", e);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    setIsStarting(true);
    try {
      await api.stopScenario();
    } catch (e) {
      console.error("Failed to stop scenario:", e);
    } finally {
      setIsStarting(false);
    }
  };

  const handleTriggerIncident = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsTriggering(true);
    try {
      if (targetCamera) {
        await api.triggerIncident({
          camera_id: targetCamera,
          severity,
          incident_type: incidentType,
        });
      } else {
        await api.triggerIncident({
          lat: parseFloat(customLat),
          lon: parseFloat(customLon),
          severity,
          incident_type: incidentType,
        });
      }
      onTriggerSuccess();
    } catch (err) {
      console.error("Error triggering incident:", err);
    } finally {
      setIsTriggering(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6 overflow-y-auto h-full">
      {/* Title */}
      <div>
        <h2 className="text-lg font-extrabold uppercase tracking-wider text-slate-100 flex items-center gap-2">
          <Zap className="w-5 h-5 text-indigo-400" />
          <span>Operational Scenario & Stress Testing Console</span>
        </h2>
        <p className="text-xs text-slate-400 font-mono mt-1">
          Configure real-world environmental conditions, inject live incidents, or simulate fleet/camera knockouts.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Scenario Selection Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4 shadow-xl">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Environmental Profile & Policy Mode</span>
          </h3>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1.5 font-bold">
                Scenario Profile
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: "normal", label: "Normal Day", icon: <Play className="w-3.5 h-3.5" /> },
                  { id: "monsoon", label: "Monsoon Surge", icon: <CloudRain className="w-3.5 h-3.5" /> },
                  { id: "ganeshotsav", label: "Ganeshotsav", icon: <AlertTriangle className="w-3.5 h-3.5" /> },
                ].map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setSelectedScenario(s.id)}
                    className={`flex flex-col items-center justify-center p-3 rounded-lg border text-xs font-mono transition-colors ${
                      selectedScenario === s.id
                        ? "bg-indigo-600/30 border-indigo-500 text-indigo-200"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <div className="mb-1">{s.icon}</div>
                    <span className="font-bold">{s.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1.5 font-bold">
                Operator Oversight Mode
              </label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  {
                    id: "assist",
                    label: "Human Assist (C2)",
                    desc: "Decisions require operator approval",
                  },
                  {
                    id: "auto",
                    label: "Autonomous Auto-Pilot",
                    desc: "Instant dispatch via CP-SAT",
                  },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedMode(m.id)}
                    className={`p-3 rounded-lg border text-left text-xs font-mono transition-colors ${
                      selectedMode === m.id
                        ? "bg-indigo-600/30 border-indigo-500 text-indigo-200"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <span className="font-bold block">{m.label}</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">{m.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="pt-2 flex items-center space-x-3">
              <button
                onClick={handleStart}
                disabled={isStarting}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 text-xs transition-colors shadow-lg shadow-indigo-950"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>APPLY & RUN SCENARIO</span>
              </button>
              <button
                onClick={handleStop}
                disabled={isStarting}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 text-xs transition-colors border border-slate-700"
              >
                <StopCircle className="w-4 h-4" />
                <span>STOP</span>
              </button>
            </div>
          </div>
        </div>

        {/* Manual Incident Injection Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4 shadow-xl">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-red-400" />
            <span>Simulate Real-Time CCTV Incident</span>
          </h3>

          <form onSubmit={handleTriggerIncident} className="space-y-3 font-mono text-xs">
            <div>
              <label className="block text-slate-400 mb-1 font-bold">Target Camera Feed</label>
              <select
                value={targetCamera}
                onChange={(e) => setTargetCamera(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="">Manual Coordinates (Map Point)</option>
                {Object.values(cameras).slice(0, 30).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id} — {c.ward_id} ({c.fov_note})
                  </option>
                ))}
              </select>
            </div>

            {!targetCamera && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-slate-400 mb-1 font-bold">Latitude</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={customLat}
                    onChange={(e) => setCustomLat(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1 font-bold">Longitude</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={customLon}
                    onChange={(e) => setCustomLon(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-slate-400 mb-1 font-bold">Incident Type</label>
                <select
                  value={incidentType}
                  onChange={(e) => setIncidentType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                >
                  <option value="accident">Accident (Traffic)</option>
                  <option value="fire_smoke">Fire / Industrial Smoke</option>
                  <option value="crowd">Crowd Surge / Panic</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1 font-bold">
                  Severity: {(severity * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={severity}
                  onChange={(e) => setSeverity(parseFloat(e.target.value))}
                  className="w-full mt-2 accent-indigo-500"
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={isTriggering}
                className="w-full bg-red-600 hover:bg-red-500 text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 text-xs transition-colors shadow-lg shadow-red-950 disabled:opacity-50"
              >
                <Flame className="w-4 h-4" />
                <span>{isTriggering ? "PROCESSING FUSION..." : "TRIGGER LIVE EMERGENCY EVENT"}</span>
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Fleet Knockout Stress Test */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2 mb-3">
          <Power className="w-4 h-4 text-cyan-400" />
          <span>Fleet Resilience & Knockout Stress Testing</span>
        </h3>
        <p className="text-xs text-slate-400 font-mono mb-4">
          Simulate random vehicle breakdowns or communication failures to evaluate system response under degraded capacity.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 font-mono text-xs">
          {Object.values(units).slice(0, 18).map((u) => {
            const isIdle = u.state === "idle" || u.state === "ready";
            return (
              <div
                key={u.id}
                className={`p-2 rounded-lg border flex flex-col justify-between ${
                  isIdle
                    ? "bg-slate-950 border-slate-800 text-slate-300"
                    : "bg-red-950/40 border-red-800/60 text-red-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-[11px] truncate">{u.id}</span>
                  <span className="text-[10px]">
                    {u.agency === "ambulance" ? "🚑" : u.agency === "police" ? "🚓" : "🚒"}
                  </span>
                </div>
                <span className="text-[9px] uppercase font-bold text-slate-400 block mb-1">
                  {u.state}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
