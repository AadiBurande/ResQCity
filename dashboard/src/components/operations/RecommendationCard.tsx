import React from "react";
import { Check, CheckCircle2, Clock, Cpu, Edit3, HelpCircle } from "lucide-react";
import { DispatchDecision, Explanation, Incident } from "../../types/schemas";

interface RecommendationCardProps {
  incident: Incident | null;
  decision: DispatchDecision | null;
  explanation: Explanation | null;
  onApprove: (decisionId: string) => void;
  onOpenOverride: (decisionId: string) => void;
  isActionLoading?: boolean;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  incident,
  decision,
  explanation,
  onApprove,
  onOpenOverride,
  isActionLoading,
}) => {
  if (!incident) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-500 font-mono text-xs">
        Select an incident from the queue or map to inspect AI dispatch recommendation.
      </div>
    );
  }

  if (!decision) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-400 font-mono text-xs">
        <Cpu className="w-6 h-6 text-indigo-400 mx-auto mb-2 animate-spin" />
        Solving joint dispatch MILP (C1) for incident {incident.id}...
      </div>
    );
  }

  const getTierBadge = (tier: string) => {
    switch (tier) {
      case "full":
        return { label: "FULL DISPATCH", bg: "bg-purple-950/60 text-purple-300 border-purple-800/80" };
      case "verify":
        return { label: "VERIFY FIRST", bg: "bg-amber-950/60 text-amber-300 border-amber-800/80" };
      default:
        return { label: "OPERATOR REVIEW", bg: "bg-rose-950/60 text-rose-300 border-rose-800/80" };
    }
  };

  const tier = getTierBadge(decision.tier);
  const isPending = decision.status === "proposed";
  const lastEta = decision.expected_last_arrival_s
    ? `${Math.floor(decision.expected_last_arrival_s / 60)}m ${Math.round(decision.expected_last_arrival_s % 60)}s`
    : "—";

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col space-y-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            CP-SAT Dispatch Recommendation
          </h3>
          <span className="font-mono text-[10px] text-slate-500">[{decision.id}]</span>
        </div>
        <div className="flex items-center space-x-2">
          <span
            className={`text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded border ${tier.bg}`}
          >
            {tier.label}
          </span>
          <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
            Policy: {decision.policy}
          </span>
        </div>
      </div>

      {/* Target & Last Arrival Telemetry */}
      <div className="grid grid-cols-3 gap-2 font-mono text-xs">
        <div className="p-2 rounded bg-slate-950 border border-slate-800">
          <span className="text-[10px] text-slate-500 block">Incident Target</span>
          <span className="text-slate-200 font-bold">{incident.ward_id}</span>
          <span className="text-[9px] text-slate-500 block">Node #{incident.node}</span>
        </div>
        <div className="p-2 rounded bg-slate-950 border border-slate-800">
          <span className="text-[10px] text-slate-500 block">Expected Last Arrival</span>
          <span className="text-indigo-400 font-bold flex items-center gap-1 mt-0.5">
            <Clock className="w-3.5 h-3.5" />
            {lastEta}
          </span>
        </div>
        <div className="p-2 rounded bg-slate-950 border border-slate-800">
          <span className="text-[10px] text-slate-500 block">Decision Status</span>
          <span
            className={`font-bold uppercase text-[11px] block mt-0.5 ${
              isPending
                ? "text-amber-400"
                : decision.status === "approved"
                ? "text-emerald-400"
                : decision.status === "overridden"
                ? "text-rose-400"
                : "text-purple-400"
            }`}
          >
            {decision.status}
          </span>
        </div>
      </div>

      {/* Assigned Units Table */}
      <div>
        <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1.5 font-bold">
          Assigned Emergency Units ({decision.assignments.length})
        </span>
        <div className="bg-slate-950 rounded-lg border border-slate-800 overflow-hidden divide-y divide-slate-800/80 font-mono text-xs">
          {decision.assignments.length === 0 ? (
            <div className="p-3 text-center text-slate-500 text-xs">
              No units assigned. Review incident or verify confidence.
            </div>
          ) : (
            decision.assignments.map((a, idx) => {
              const etaFormatted = `${Math.floor(a.eta_s / 60)}m ${Math.round(a.eta_s % 60)}s`;
              return (
                <div key={idx} className="p-2 flex items-center justify-between text-slate-300">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs">
                      {a.unit_id.startsWith("AMB")
                        ? "🚑"
                        : a.unit_id.startsWith("POL")
                        ? "🚓"
                        : "🚒"}
                    </span>
                    <span className="font-bold text-slate-200">{a.unit_id}</span>
                    <span className="text-[10px] text-slate-500">({a.role})</span>
                  </div>
                  <div className="flex items-center space-x-3 text-right">
                    {a.hospital_id && (
                      <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/40">
                        → {a.hospital_id}
                      </span>
                    )}
                    <span className="text-emerald-400 font-bold">{etaFormatted}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Explainability & Factor Bars */}
      {explanation && (
        <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 space-y-2">
          <div className="flex items-center space-x-1.5 text-slate-300 text-xs">
            <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-bold font-mono text-[11px] uppercase">
              Model Justification
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed font-sans">{explanation.text}</p>

          <div className="pt-2 space-y-1.5 font-mono text-[10px]">
            {explanation.factors.map((f, i) => (
              <div key={i} className="space-y-0.5">
                <div className="flex justify-between text-slate-400">
                  <span>{f.name}</span>
                  <span className="text-slate-200 font-bold">
                    {(f.value * 100).toFixed(0)}% (w={f.weight.toFixed(2)})
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                  <div
                    className="bg-indigo-400 h-1 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, f.value * 100))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Operator Action Bar */}
      {isPending && (
        <div className="pt-1 flex items-center space-x-3 select-none">
          <button
            onClick={() => onApprove(decision.id)}
            disabled={isActionLoading}
            className="flex-1 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 text-xs transition-colors shadow-lg shadow-emerald-900/40 disabled:opacity-50"
          >
            <Check className="w-4 h-4 stroke-[3]" />
            <span>APPROVE DISPATCH [A]</span>
          </button>
          <button
            onClick={() => onOpenOverride(decision.id)}
            disabled={isActionLoading}
            className="flex-1 bg-amber-600/90 hover:bg-amber-500 active:bg-amber-700 text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 text-xs transition-colors shadow-lg shadow-amber-900/40 disabled:opacity-50"
          >
            <Edit3 className="w-4 h-4" />
            <span>OVERRIDE [O]</span>
          </button>
        </div>
      )}

      {!isPending && (
        <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-center font-mono text-xs text-slate-400 flex items-center justify-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>Decision recorded · Audit logged with cryptographic trace</span>
        </div>
      )}
    </div>
  );
};
