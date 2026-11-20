import React, { useEffect, useRef } from "react";
import L from "leaflet";
import { Camera, Hospital, Incident, Station, Unit } from "../../types/schemas";

interface PuneMapProps {
  cameras: Record<string, Camera>;
  stations: Record<string, Station>;
  hospitals: Record<string, Hospital>;
  units: Record<string, Unit>;
  incidents: Record<string, Incident>;
  selectedIncidentId: string | null;
  onSelectIncident: (id: string) => void;
  onSelectCamera: (id: string) => void;
  onMapClick?: (lat: number, lon: number) => void;
}

export const PuneMap: React.FC<PuneMapProps> = ({
  cameras,
  stations,
  hospitals,
  units,
  incidents,
  selectedIncidentId,
  onSelectIncident,
  onSelectCamera,
  onMapClick,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerGroupsRef = useRef<{
    stations: L.LayerGroup;
    hospitals: L.LayerGroup;
    cameras: L.LayerGroup;
    units: L.LayerGroup;
    incidents: L.LayerGroup;
    routes: L.LayerGroup;
  }>({
    stations: L.layerGroup(),
    hospitals: L.layerGroup(),
    cameras: L.layerGroup(),
    units: L.layerGroup(),
    incidents: L.layerGroup(),
    routes: L.layerGroup(),
  });

  const onMapClickRef = useRef(onMapClick);
  onMapClickRef.current = onMapClick;

  const onSelectIncidentRef = useRef(onSelectIncident);
  onSelectIncidentRef.current = onSelectIncident;

  const onSelectCameraRef = useRef(onSelectCamera);
  onSelectCameraRef.current = onSelectCamera;

  // Initialize Map exactly once
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [18.5204, 73.8567],
      zoom: 13,
      zoomControl: false,
      attributionControl: false,
    });

    L.control.zoom({ position: "bottomright" }).addTo(map);

    // OpenStreetMap dark styled tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
    }).addTo(map);

    // Add layer groups
    const lg = layerGroupsRef.current;
    lg.stations.addTo(map);
    lg.hospitals.addTo(map);
    lg.cameras.addTo(map);
    lg.routes.addTo(map);
    lg.units.addTo(map);
    lg.incidents.addTo(map);

    map.on("click", (e: L.LeafletMouseEvent) => {
      onMapClickRef.current?.(e.latlng.lat, e.latlng.lng);
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update Stations & Hospitals
  useEffect(() => {
    const lg = layerGroupsRef.current;
    lg.stations.clearLayers();
    lg.hospitals.clearLayers();

    // Stations
    Object.values(stations).forEach((stn) => {
      const color =
        stn.agency === "ambulance" ? "#10b981" : stn.agency === "police" ? "#3b82f6" : "#ef4444";
      const icon = L.divIcon({
        className: "custom-station-icon",
        html: `<div style="background-color:${color};width:10px;height:10px;border-radius:2px;border:1px solid #fff;box-shadow:0 0 4px ${color};"></div>`,
        iconSize: [10, 10],
      });
      L.marker([stn.location.lat, stn.location.lon], { icon })
        .bindTooltip(`<b>${stn.name}</b><br><span style="text-transform:uppercase">${stn.agency}</span> Station`, {
          direction: "top",
          className: "leaflet-dark-tooltip",
        })
        .addTo(lg.stations);
    });

    // Hospitals
    Object.values(hospitals).forEach((hosp) => {
      const icon = L.divIcon({
        className: "custom-hospital-icon",
        html: `<div style="background-color:#06b6d4;width:12px;height:12px;border-radius:50%;border:2px solid #fff;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:bold;color:#000;">H</div>`,
        iconSize: [12, 12],
      });
      L.marker([hosp.location.lat, hosp.location.lon], { icon })
        .bindTooltip(`<b>${hosp.name}</b><br>ICU Beds: ${hosp.beds_available}/${hosp.icu_beds}`, {
          direction: "top",
          className: "leaflet-dark-tooltip",
        })
        .addTo(lg.hospitals);
    });
  }, [stations, hospitals]);

  // Update Cameras
  useEffect(() => {
    const lg = layerGroupsRef.current;
    lg.cameras.clearLayers();

    Object.values(cameras).forEach((cam) => {
      const icon = L.divIcon({
        className: "custom-camera-icon",
        html: `<div style="background-color:#475569;width:8px;height:8px;border-radius:50%;border:1px solid #94a3b8;" title="${cam.id}"></div>`,
        iconSize: [8, 8],
      });
      const marker = L.marker([cam.lat, cam.lon], { icon }).addTo(lg.cameras);
      marker.on("click", () => onSelectCameraRef.current(cam.id));
      marker.bindTooltip(`CCTV: <b>${cam.id}</b> (${cam.ward_id})`, { direction: "top" });
    });
  }, [cameras]);

  // Update Incidents
  useEffect(() => {
    const lg = layerGroupsRef.current;
    lg.incidents.clearLayers();

    Object.values(incidents).forEach((inc) => {
      const isSelected = selectedIncidentId === inc.id;
      const pulseColor = inc.severity >= 0.7 ? "#ef4444" : "#f59e0b";

      const icon = L.divIcon({
        className: "custom-incident-icon",
        html: `
          <div style="position:relative;width:24px;height:24px;display:flex;align-items:center;justify-content:center;">
            <div class="pulsing-ring" style="width:24px;height:24px;border:2px solid ${pulseColor};background-color:${pulseColor}33;"></div>
            <div style="background-color:${pulseColor};width:14px;height:14px;border-radius:50%;border:2px solid #fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 10px ${pulseColor};font-size:8px;">⚠️</div>
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      const marker = L.marker([inc.location.lat, inc.location.lon], { icon }).addTo(lg.incidents);
      marker.on("click", () => onSelectIncidentRef.current(inc.id));
      marker.bindTooltip(
        `<b>${inc.type.toUpperCase()}</b><br>Ward: ${inc.ward_id}<br>Posterior: ${(inc.posterior * 100).toFixed(0)}% · Severity: ${(inc.severity * 100).toFixed(0)}%`,
        { direction: "top", permanent: isSelected }
      );
    });
  }, [incidents, selectedIncidentId]);

  // Update Units
  useEffect(() => {
    const lg = layerGroupsRef.current;
    lg.units.clearLayers();

    Object.values(units).forEach((u) => {
      if (!u.position) return;
      const color =
        u.agency === "ambulance" ? "#10b981" : u.agency === "police" ? "#3b82f6" : "#ef4444";
      const iconChar = u.agency === "ambulance" ? "🚑" : u.agency === "police" ? "🚓" : "🚒";
      const isMoving = ["en_route", "transporting", "returning"].includes(u.state);

      const icon = L.divIcon({
        className: "custom-unit-icon",
        html: `
          <div style="background-color:${color};width:20px;height:20px;border-radius:50%;border:1.5px solid #fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 ${isMoving ? "8px" : "3px"} ${color};font-size:10px;">
            ${iconChar}
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });

      L.marker([u.position.lat, u.position.lon], { icon })
        .bindTooltip(
          `<b>${u.id}</b><br>Agency: ${u.agency}<br>Status: <b style="text-transform:uppercase">${u.state}</b>`,
          { direction: "top" }
        )
        .addTo(lg.units);
    });
  }, [units]);

  return (
    <div className="relative w-full h-full bg-slate-950">
      <div ref={mapContainerRef} className="w-full h-full" />
      {/* Legend overlay */}
      <div className="absolute bottom-4 left-4 z-[500] bg-slate-900/90 backdrop-blur-sm border border-slate-800 rounded-lg p-2.5 text-[11px] font-mono flex flex-col space-y-1.5 shadow-xl select-none">
        <span className="text-slate-400 font-bold uppercase text-[9px] tracking-wider mb-0.5">
          Map Legend
        </span>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span className="text-slate-300">Ambulance (108)</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
          <span className="text-slate-300">Police Patrol</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
          <span className="text-slate-300">Fire Engine</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
          <span className="text-slate-300">Trauma Hospital</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs">⚠️</span>
          <span className="text-slate-300">Active Incident</span>
        </div>
      </div>
    </div>
  );
};
