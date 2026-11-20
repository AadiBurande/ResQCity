import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, CloudRain, Shield, Wifi, WifiOff } from "lucide-react";
import { ScenarioStatus } from "../api/client";

interface HeaderProps {
  scenarioStatus: ScenarioStatus;
  connected: boolean;
  activeIncidentsCount: number;
  lastNotification: string | null;
}

export const Header: React.FC<HeaderProps> = ({
  scenarioStatus,
  connected,
  activeIncidentsCount,
  lastNotification,
}) => {
  const [time, setTime] = useState(new Date().toUTCString());

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC");
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-4 py-2.5 flex items-center justify-between select-none">
      {/* Brand & Title */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-red-600/20 text-red-400 border border-red-500/30">
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-extrabold text-base tracking-wider text-slate-100 flex items-center gap-1.5">
              <span>RESQCITY</span>
              <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono font-medium border border-indigo-500/30">
                PUNE EOC
              </span>
            </h1>
          </div>
          <p className="text-[10px] text-slate-400 font-mono leading-none mt-0.5">
            CCTV-Triggered Joint Multi-Agency Dispatch System
          </p>
        </div>
      </div>

      {/* Live notification ticker */}
      <div className="hidden lg:flex items-center bg-slate-950/70 border border-slate-800/80 rounded-full px-4 py-1 max-w-md">
        <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse mr-2 shrink-0" />
        <span className="text-xs font-mono text-slate-300 truncate">
          {lastNotification || "System active · Monitoring 60 CCTV streams across 15 wards"}
        </span>
      </div>

      {/* Status Badges & Controls */}
      <div className="flex items-center space-x-3 text-xs">
        {/* Scenario badge */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
          {scenarioStatus.rain_mm_h > 0 ? (
            <CloudRain className="w-3.5 h-3.5 text-blue-400" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          )}
          <span className="text-slate-400 font-medium">Scenario:</span>
          <span className="font-mono uppercase font-bold text-slate-200">
            {scenarioStatus.scenario}
          </span>
          {scenarioStatus.rain_mm_h > 0 && (
            <span className="text-blue-400 font-mono text-[10px]">
              ({scenarioStatus.rain_mm_h.toFixed(1)} mm/h)
            </span>
          )}
        </div>

        {/* Dispatch Mode badge */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
          <span className="text-slate-400 font-medium">Mode:</span>
          <span
            className={`font-mono font-bold uppercase ${
              scenarioStatus.mode === "auto" ? "text-purple-400" : "text-emerald-400"
            }`}
          >
            {scenarioStatus.mode === "auto" ? "Auto-Pilot" : "Human Assist (C2)"}
          </span>
        </div>

        {/* Active Incidents */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
          <span className="text-slate-400 font-medium">Incidents:</span>
          <span className="font-mono font-bold text-red-400">{activeIncidentsCount}</span>
        </div>

        {/* Connection status */}
        <div
          className={`flex items-center space-x-1.5 px-2 py-1 rounded border ${
            connected
              ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-400"
              : "bg-red-950/40 border-red-800/60 text-red-400"
          }`}
        >
          {connected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
          <span className="font-mono text-[11px]">{connected ? "ONLINE" : "OFFLINE"}</span>
        </div>

        {/* Clock */}
        <div className="hidden sm:block font-mono text-slate-400 text-xs px-2">
          {time}
        </div>
      </div>
    </header>
  );
};
