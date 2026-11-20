import React from "react";
import { Camera as CameraIcon, Lock, Video } from "lucide-react";
import { Camera } from "../../types/schemas";

interface CameraFeedTileProps {
  camera: Camera | null;
  hasIncident?: boolean;
}

export const CameraFeedTile: React.FC<CameraFeedTileProps> = ({ camera, hasIncident }) => {
  if (!camera) {
    return (
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 text-center text-slate-500 font-mono text-xs">
        <Video className="w-5 h-5 mx-auto mb-1 text-slate-600" />
        Click a camera pin on the map to view edge-blurred live stream.
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden shadow-lg">
      <div className="bg-slate-950 px-3 py-1.5 border-b border-slate-800 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-2">
          <CameraIcon className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-bold text-slate-200">{camera.id}</span>
          <span className="text-slate-500">({camera.ward_id})</span>
        </div>
        <div className="flex items-center space-x-1.5 text-[10px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/40">
          <Lock className="w-3 h-3" />
          <span>DPDP BLURRED</span>
        </div>
      </div>

      {/* Simulated camera feed canvas / graphic */}
      <div className="relative aspect-video bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center overflow-hidden">
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b15_1px,transparent_1px),linear-gradient(to_bottom,#1e293b15_1px,transparent_1px)] bg-[size:16px_16px]" />

        {/* Live indicator */}
        <div className="absolute top-2 left-2 z-10 flex items-center space-x-1.5 px-2 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] font-mono text-white">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span>REC · 25 FPS</span>
        </div>

        {/* Simulated detection bounding box if incident active */}
        {hasIncident ? (
          <div className="relative border-2 border-red-500/80 bg-red-500/10 p-3 rounded text-center backdrop-blur-[2px]">
            <span className="bg-red-600 text-white font-mono font-bold text-[9px] px-1.5 py-0.5 rounded absolute -top-3 left-1">
              ACCIDENT · 88%
            </span>
            <span className="font-mono text-xs text-red-300 font-bold">
              [VEHICLE COLLISION DETECTED]
            </span>
            <div className="text-[10px] text-slate-400 mt-1">
              Faces & plates blurred at edge (D-026)
            </div>
          </div>
        ) : (
          <div className="text-center font-mono text-slate-600 text-xs">
            <span>LIVE FEED: {camera.fov_note}</span>
            <div className="text-[10px] text-slate-700 mt-1">Normal Traffic Flow</div>
          </div>
        )}
      </div>
    </div>
  );
};
