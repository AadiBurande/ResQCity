import React from "react";
import { AlertCircle, Flame, ShieldAlert, Users } from "lucide-react";
import { Incident } from "../../types/schemas";

interface IncidentQueueProps {
  incidents: Incident[];
  selectedIncidentId: string | null;
  onSelectIncident: (id: string) => void;
}

export const IncidentQueue: React.FC<IncidentQueueProps> = ({
  incidents,
  selectedIncidentId,
  onSelectIncident,
}) => {
  // Sort by priority: severity * posterior descending
  const sorted = [...incidents].sort((a, b) => {
    const prioA = a.severity * a.posterior;
    const prioB = b.severity * b.posterior;
    return prioB - prioA;
  });

  const getIcon = (type: string) => {
    switch (type) {
      case "fire_smoke":
        return <Flame className="w-4 h-4 text-red-400" />;
      case "crowd":
        return <Users className="w-4 h-4 text-amber-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-orange-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "confirmed":
        return "bg-amber-500/20 text-amber-400 border-amber-500/30";
      case "dispatched":
        return "bg-indigo-500/20 text-indigo-400 border-indigo-500/30";
      case "on_scene":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "resolved":
        return "bg-slate-500/20 text-slate-400 border-slate-500/30";
      default:
        return "bg-slate-700 text-slate-300";
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 w-80 shrink-0">
      <div className="px-3.5 py-2.5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-red-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Incident Queue
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
          {sorted.length} Active
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {sorted.length === 0 ? (
          <div className="text-center py-10 px-4 text-slate-500 text-xs font-mono">
            No active incidents.<br />All wards clear.
          </div>
        ) : (
          sorted.map((inc) => {
            const isSelected = selectedIncidentId === inc.id;
            const prio = inc.severity * inc.posterior;

            return (
              <div
                key={inc.id}
                onClick={() => onSelectIncident(inc.id)}
                className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-slate-800/90 border-indigo-500 shadow-md shadow-indigo-500/10"
                    : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center space-x-1.5">
                    {getIcon(inc.type)}
                    <span className="font-mono text-xs font-bold uppercase text-slate-200">
                      {inc.id}
                    </span>
                  </div>
                  <span
                    className={`text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded border ${getStatusBadge(
                      inc.status
                    )}`}
                  >
                    {inc.status}
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2">
                  <span className="font-medium text-slate-300">{inc.ward_id}</span>
                  <span className="font-mono text-[10px] text-slate-500">
                    Prio: {(prio * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Posterior & Severity Bars */}
                <div className="space-y-1 mb-2 font-mono text-[10px]">
                  <div className="flex justify-between text-slate-400">
                    <span>Bayesian Posterior:</span>
                    <span className="text-slate-200 font-bold">
                      {(inc.posterior * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-1 rounded-full transition-all"
                      style={{ width: `${Math.min(100, inc.posterior * 100)}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-slate-400">
                    <span>Severity:</span>
                    <span className="text-amber-400 font-bold">
                      {(inc.severity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                    <div
                      className="bg-amber-500 h-1 rounded-full transition-all"
                      style={{ width: `${Math.min(100, inc.severity * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Requirements badges */}
                <div className="flex items-center space-x-1 text-[10px] font-mono">
                  {inc.requirements.ambulance > 0 && (
                    <span className="px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/50">
                      🚑 {inc.requirements.ambulance}
                    </span>
                  )}
                  {inc.requirements.police > 0 && (
                    <span className="px-1.5 py-0.5 rounded bg-blue-950/60 text-blue-400 border border-blue-800/50">
                      🚓 {inc.requirements.police}
                    </span>
                  )}
                  {inc.requirements.fire > 0 && (
                    <span className="px-1.5 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/50">
                      🚒 {inc.requirements.fire}
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
