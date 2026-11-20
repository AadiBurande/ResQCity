import React, { useCallback, useState } from "react";
import { Header } from "./components/Header";
import { Navigation, TabId } from "./components/Navigation";
import { PuneMap } from "./components/map/PuneMap";
import { IncidentQueue } from "./components/operations/IncidentQueue";
import { RecommendationCard } from "./components/operations/RecommendationCard";
import { OverrideModal } from "./components/operations/OverrideModal";
import { CameraFeedTile } from "./components/operations/CameraFeedTile";
import { ScenarioManager } from "./components/scenario/ScenarioManager";
import { KpiDashboard } from "./components/kpi/KpiDashboard";
import { ResultsViewer } from "./components/results/ResultsViewer";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useResQState } from "./hooks/useResQState";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { api } from "./api/client";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("operations");
  const [isOverrideModalOpen, setIsOverrideModalOpen] = useState(false);
  const [isActionLoading, setIsActionLoading] = useState(false);

  const {
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
  } = useResQState();

  // Find currently selected incident & decision
  const activeIncidentList = Object.values(incidents);
  const selectedIncident =
    (selectedIncidentId && incidents[selectedIncidentId]) ||
    activeIncidentList[0] ||
    null;

  const selectedDecision = selectedIncident
    ? Object.values(decisions).find((d) => d.incident_id === selectedIncident.id) || null
    : null;

  const selectedExplanation = selectedDecision ? explanations[selectedDecision.id] || null : null;

  const selectedCamera =
    (selectedCameraId && cameras[selectedCameraId]) ||
    (selectedIncident && cameras[selectedIncident.sources?.[0]?.ref_id]) ||
    Object.values(cameras)[0] ||
    null;

  // Pending decisions count for badge
  const pendingDecisionsCount = Object.values(decisions).filter(
    (d) => d.status === "proposed"
  ).length;

  // Actions
  const handleApprove = async (decisionId: string) => {
    setIsActionLoading(true);
    try {
      await api.approveDecision(decisionId);
      await refreshStatic();
    } catch (e) {
      console.error("Failed to approve decision:", e);
    } finally {
      setIsActionLoading(false);
    }
  };

  const handleOpenOverride = (_decisionId?: string) => {
    setIsOverrideModalOpen(true);
  };

  const handleCommitOverride = async (
    decisionId: string,
    reason: string,
    updatedAssignments: any[]
  ) => {
    setIsActionLoading(true);
    try {
      await api.overrideDecision(decisionId, reason, updatedAssignments);
      setIsOverrideModalOpen(false);
      await refreshStatic();
    } catch (e) {
      console.error("Failed to override decision:", e);
    } finally {
      setIsActionLoading(false);
    }
  };

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onApprove: () => {
      if (selectedDecision && selectedDecision.status === "proposed") {
        handleApprove(selectedDecision.id);
      }
    },
    onOverride: () => {
      if (selectedDecision && selectedDecision.status === "proposed") {
        setIsOverrideModalOpen(true);
      }
    },
    onTabChange: (tab) => setActiveTab(tab),
    onEscape: () => setIsOverrideModalOpen(false),
  });

  const handleMapClick = useCallback((lat: number, lon: number) => {
    console.log("Map clicked at", lat, lon);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 overflow-hidden font-sans">
      {/* Header */}
      <Header
        scenarioStatus={scenarioStatus}
        connected={connected}
        activeIncidentsCount={activeIncidentList.length}
        lastNotification={lastNotification}
      />

      {/* Navigation tabs */}
      <Navigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
        pendingDecisionsCount={pendingDecisionsCount}
      />

      {/* Tab Contents */}
      <main className="flex-1 relative overflow-hidden">
        <ErrorBoundary fallbackTitle="Operations View Error">
          {activeTab === "operations" && (
            <div className="flex h-full w-full">
              {/* Left: Incident Queue */}
              <IncidentQueue
                incidents={activeIncidentList}
                selectedIncidentId={selectedIncident?.id || null}
                onSelectIncident={(id) => {
                  setSelectedIncidentId(id);
                  const inc = incidents[id];
                  if (inc && inc.sources?.[0]?.ref_id && cameras[inc.sources[0].ref_id]) {
                    setSelectedCameraId(inc.sources[0].ref_id);
                  }
                }}
              />

              {/* Center: Map */}
              <div className="flex-1 h-full relative">
                <PuneMap
                  cameras={cameras}
                  stations={stations}
                  hospitals={hospitals}
                  units={units}
                  incidents={incidents}
                  selectedIncidentId={selectedIncident?.id || null}
                  onSelectIncident={(id) => setSelectedIncidentId(id)}
                  onSelectCamera={(id) => setSelectedCameraId(id)}
                  onMapClick={handleMapClick}
                />
              </div>

              {/* Right: Decision Recommendation & Camera Preview */}
              <div className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col p-3 space-y-3 overflow-y-auto shrink-0 shadow-2xl">
                <RecommendationCard
                  incident={selectedIncident}
                  decision={selectedDecision}
                  explanation={selectedExplanation}
                  onApprove={handleApprove}
                  onOpenOverride={handleOpenOverride}
                  isActionLoading={isActionLoading}
                />

                <CameraFeedTile
                  camera={selectedCamera}
                  hasIncident={
                    selectedIncident &&
                    selectedCamera &&
                    selectedIncident.sources?.some((s) => s.ref_id === selectedCamera.id)
                  }
                />
              </div>
            </div>
          )}

          {activeTab === "scenario" && (
            <ScenarioManager
              scenarioStatus={scenarioStatus}
              cameras={cameras}
              units={units}
              onTriggerSuccess={() => {
                setActiveTab("operations");
                refreshStatic();
              }}
            />
          )}

          {activeTab === "kpis" && <KpiDashboard kpis={kpis} />}

          {activeTab === "results" && <ResultsViewer />}
        </ErrorBoundary>
      </main>

      {/* Override Modal */}
      {selectedDecision && (
        <OverrideModal
          decision={selectedDecision}
          isOpen={isOverrideModalOpen}
          onClose={() => setIsOverrideModalOpen(false)}
          onSubmit={handleCommitOverride}
          isSubmitting={isActionLoading}
        />
      )}
    </div>
  );
};
