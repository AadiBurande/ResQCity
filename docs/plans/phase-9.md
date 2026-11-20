# Phase 9 — Control-Room Dashboard & Interactive Viva Console

Status: implemented · Date: 2026-11-20 · Decisions: docs/decisions.md

## Goal

Build an operator-grade, accessible control-room dashboard in `dashboard/` (React, Vite, TypeScript, Leaflet) for live multi-agency emergency operations in Pune. The dashboard provides:
1. **Live Operations:** Leaflet map with Pune road network context, facilities, moving emergency units, CCTV feeds, incident queue, recommendation card with explainability factor bars, and keyboard-driven Approve/Override actions.
2. **Scenario Control:** Live switching of environmental conditions (Normal, Monsoon, Ganeshotsav), operational modes (Assist vs. Auto), manual incident injection, and unit knockout stress tests.
3. **KPIs & Telemetry:** Real-time gauges of fleet utilisation, detection-to-dispatch latency, response time quantiles (p50/p90), and operator override rate.
4. **Research Results Viewer:** A dedicated defense/viva screen displaying the 5 research figures, statistical summary tables, and hypothesis validation from Phase 10.

## Architectural Boundaries

- Frontend lives strictly in `dashboard/`.
- Types are synchronised from `src/resqcity/common/schemas.py` into `dashboard/src/types/schemas.ts` via `uv run python -m resqcity.tools.gen_ts_types`.
- Communicates with FastAPI backend (`src/resqcity/backend/`) via REST endpoints (`/api/...`) and real-time WebSocket (`/ws`).
- When backend is offline or during standalone UI development, a mock mode allows the interface to run seamlessly.

## Design Principles

- **Control-Room Ergonomics:** High-contrast dark theme (Slate-950/Slate-900), monospace telemetry readouts, clear visual hierarchy.
- **Accessible Multi-Modal Encoding:** Color is never used alone; agency badges feature distinctive icons (Ambulance 🚑, Police 🚓, Fire 🚒) and decision tiers have explicit textual labels (`FULL`, `VERIFY`, `REVIEW`).
- **Keyboard Shortcuts:** Fast operator workflows via keyboard (`A` for Approve, `O` for Override, `Esc` to close modal, `1`-`4` to switch views).
- **Explainability First:** Decisions display plain-English justifications alongside quantitative factor bars showing relative influence (Bayesian posterior, severity, unit proximity).

## System Architecture

```
dashboard/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/
│   ├── types/
│   │   └── schemas.ts          (Auto-generated shared models)
│   ├── api/
│   │   ├── client.ts           (REST client with fetch & error handling)
│   │   └── websocket.ts        (Reconnecting WebSocket event subscriber)
│   ├── components/
│   │   ├── Header.tsx          (Global status, scenario switcher, time)
│   │   ├── Navigation.tsx      (Tab switcher: Operations, Scenario, KPIs, Results)
│   │   ├── map/
│   │   │   ├── PuneMap.tsx     (Leaflet container, layers, dark tiles)
│   │   │   ├── UnitMarker.tsx  (Moving vehicle markers with direction & state)
│   │   │   ├── IncidentMarker.tsx (Pulsing accident/fire markers)
│   │   │   └── FacilityMarker.tsx (Stations & Hospitals)
│   │   ├── operations/
│   │   │   ├── IncidentQueue.tsx   (Active incidents ranked by severity × posterior)
│   │   │   ├── RecommendationCard.tsx (Dispatch decision, tier, factor bars)
│   │   │   ├── OverrideModal.tsx   (Manual vehicle reassignment & reason capture)
│   │   │   └── CameraFeedTile.tsx  (CCTV thumbnail with privacy blur)
│   │   ├── scenario/
│   │   │   ├── ScenarioManager.tsx (Weather/festival profiles)
│   │   │   ├── IncidentInjector.tsx (Point-and-click accident creation)
│   │   │   └── FleetKnockout.tsx   (Disable units for resilience testing)
│   │   ├── kpi/
│   │   │   └── KpiDashboard.tsx    (Real-time charts, quantiles, utilisations)
│   │   └── results/
│   │       └── ResultsViewer.tsx   (Phase 10 figures, tables, hypothesis cards)
│   ├── hooks/
│   │   ├── useResQState.ts     (Central state syncing REST + WS)
│   │   └── useKeyboardShortcuts.ts (Operator hotkeys)
│   ├── styles/
│   │   └── globals.css         (Tailwind / high-contrast CSS tokens)
│   ├── App.tsx
│   └── main.tsx
└── tests/
    └── e2e.spec.ts             (End-to-end integration test)
```

## Implementation Phases

1. **Setup & Tooling:**
   - Configure `package.json`, `vite.config.ts`, `tsconfig.json`, Tailwind/CSS.
   - Install dependencies (`leaflet`, `lucide-react`, `chart.js` or SVG sparklines, etc.).
2. **API & State Integration:**
   - Implement REST API client (`/cameras`, `/units`, `/stations`, `/hospitals`, `/incidents`, `/decisions`, `/kpis`).
   - Implement WebSocket subscriber for real-time bus events (`units_tick`, `incident_update`, `decision_ready`, etc.).
3. **Core Control-Room UI:**
   - Implement Live Map with dark tiles, unit tracking, incident markers.
   - Implement Incident Queue and Recommendation Card with factor bars.
   - Implement Approve & Override workflows with hotkeys (`A` / `O`).
4. **Scenario Control & KPIs:**
   - Implement Scenario Switcher, Incident Injector, Fleet Knockout.
   - Implement KPI dashboard cards and gauge readouts.
5. **Research Results Screen:**
   - Integrate the 5 Phase 10 figures and summary findings table.
6. **Integration & Task Runner:**
   - Wire `ui` and `demo` in `src/resqcity/tools/tasks.py`.
   - Validate full flow via automated test.
