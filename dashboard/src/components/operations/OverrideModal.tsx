import React, { useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { DispatchDecision } from "../../types/schemas";

interface OverrideModalProps {
  decision: DispatchDecision;
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (decisionId: string, reason: string, updatedAssignments: any[]) => void;
  isSubmitting?: boolean;
}

export const OverrideModal: React.FC<OverrideModalProps> = ({
  decision,
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
}) => {
  const [reason, setReason] = useState("");
  const [assignments, setAssignments] = useState(decision.assignments);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim()) return;
    onSubmit(decision.id, reason, assignments);
  };

  const removeUnit = (index: number) => {
    setAssignments((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="fixed inset-0 z-[1000] bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-lg shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-5 py-3.5 bg-slate-800/80 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-amber-400 font-bold text-sm">
            <AlertTriangle className="w-4 h-4" />
            <span>OPERATOR DISPATCH OVERRIDE</span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-700"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div className="text-xs text-slate-300">
            Per DPDP & C2 protocol, all human overrides are permanently logged to the legal audit
            trail with operator identity, timestamp, and justification.
          </div>

          <div>
            <label className="block text-xs font-mono font-bold uppercase text-slate-400 mb-1">
              Justification / Reason (Mandatory)
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Local waterlogging at Swargate junction; re-routing via Tilak Road..."
              rows={3}
              required
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 font-sans"
            />
          </div>

          {/* Unit list */}
          <div>
            <label className="block text-xs font-mono font-bold uppercase text-slate-400 mb-1">
              Adjust Dispatched Fleet
            </label>
            <div className="space-y-1.5 font-mono text-xs">
              {assignments.map((a, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-2 rounded bg-slate-950 border border-slate-800"
                >
                  <span className="text-slate-200 font-bold">{a.unit_id}</span>
                  <div className="flex items-center space-x-3">
                    <span className="text-slate-400 text-[11px]">{a.eta_s}s</span>
                    <button
                      type="button"
                      onClick={() => removeUnit(i)}
                      className="text-red-400 hover:text-red-300 text-[10px] uppercase font-bold"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="pt-2 flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800"
            >
              Cancel [Esc]
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !reason.trim()}
              className="px-5 py-2 rounded-lg text-xs font-bold bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-900/30 disabled:opacity-50"
            >
              {isSubmitting ? "Logging..." : "Confirm & Commit Override"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
