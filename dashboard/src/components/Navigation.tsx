import React from "react";
import { BarChart3, BookOpen, Compass, Sliders } from "lucide-react";

export type TabId = "operations" | "scenario" | "kpis" | "results";

interface NavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  pendingDecisionsCount: number;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  onTabChange,
  pendingDecisionsCount,
}) => {
  const tabs: Array<{ id: TabId; label: string; icon: React.ReactNode; shortcut: string; badge?: number }> = [
    {
      id: "operations",
      label: "Live Operations",
      icon: <Compass className="w-4 h-4" />,
      shortcut: "1",
      badge: pendingDecisionsCount > 0 ? pendingDecisionsCount : undefined,
    },
    {
      id: "scenario",
      label: "Scenario & Stress Test",
      icon: <Sliders className="w-4 h-4" />,
      shortcut: "2",
    },
    {
      id: "kpis",
      label: "Telemetry & KPIs",
      icon: <BarChart3 className="w-4 h-4" />,
      shortcut: "3",
    },
    {
      id: "results",
      label: "Research Findings (Viva)",
      icon: <BookOpen className="w-4 h-4" />,
      shortcut: "4",
    },
  ];

  return (
    <nav className="bg-slate-900/90 border-b border-slate-800 px-4 flex items-center space-x-1 select-none">
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center space-x-2 px-3.5 py-2 text-xs font-medium border-b-2 transition-colors relative ${
              isActive
                ? "border-indigo-500 text-indigo-400 bg-slate-800/60"
                : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
            <span className="text-[10px] text-slate-500 font-mono px-1 rounded bg-slate-800">
              [{tab.shortcut}]
            </span>
            {tab.badge !== undefined && (
              <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-mono font-bold bg-amber-500 text-slate-950 animate-bounce">
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
};
