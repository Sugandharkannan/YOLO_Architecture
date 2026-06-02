import React from 'react';

interface ModelSpec {
  name: string;
  params: number;
  flops: number; // in Billions (GFLOPs)
  speed: number; // ms on GPU
  map: number; // mAP50-95
  desc: string;
  idealFor: string;
}

const MODEL_SPECS: ModelSpec[] = [
  {
    name: 'YOLOv8n (Nano)',
    params: 3200000,
    flops: 8.7,
    speed: 1.2,
    map: 37.3,
    desc: 'Lightest model. Highly optimized for mobile devices and edge processors.',
    idealFor: 'Edge devices, Raspberry Pi, Mobile Apps'
  },
  {
    name: 'YOLOv8s (Small)',
    params: 11200000,
    flops: 28.6,
    speed: 2.0,
    map: 44.9,
    desc: 'Balanced model. Provides excellent speed with double-digit accuracy gains.',
    idealFor: 'Real-time servers, desktop webcams, IoT hubs'
  },
  {
    name: 'YOLOv8m (Medium)',
    params: 25900000,
    flops: 78.9,
    speed: 3.7,
    map: 50.2,
    desc: 'Heavy model. Resolves highly cluttered objects and fine textures.',
    idealFor: 'AI research, offline batch analysis, high-end GPUs'
  }
];

export const ModelComparer: React.FC = () => {
  // Finds the maximum value for normalization in progress bars
  const maxParams = Math.max(...MODEL_SPECS.map(m => m.params));
  const maxFlops = Math.max(...MODEL_SPECS.map(m => m.flops));
  const maxSpeed = Math.max(...MODEL_SPECS.map(m => m.speed));
  const maxMap = 60.0; // Benchmark max reference for YOLOv8 family

  return (
    <div className="flex flex-col space-y-4">
      {/* Title */}
      <div>
        <h3 className="font-semibold text-lg text-slate-100">
          Model Comparison Panel
        </h3>
        <p className="text-xs text-slate-400">
          Compare YOLOv8 variants across efficiency, computational density, and accuracy metrics.
        </p>
      </div>

      {/* Model Spec Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {MODEL_SPECS.map((model) => (
          <div
            key={model.name}
            className="border border-white/5 bg-slate-900/40 rounded-xl p-4 flex flex-col space-y-3 relative overflow-hidden"
          >
            {/* Visual Header */}
            <div className="flex items-center justify-between">
              <span className="font-semibold text-sm text-indigo-400 tracking-wide">
                {model.name}
              </span>
              <span className="text-[10px] bg-slate-800/80 text-slate-300 font-mono px-2 py-0.5 rounded">
                v8 Family
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed min-h-[40px]">
              {model.desc}
            </p>

            <div className="border-t border-white/5 pt-3 space-y-2.5">
              {/* Parameter Bar */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-slate-500">Parameters</span>
                  <span className="text-slate-300">{(model.params / 1e6).toFixed(1)}M</span>
                </div>
                <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                    style={{ width: `${(model.params / maxParams) * 100}%` }}
                  />
                </div>
              </div>

              {/* FLOPs Bar */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-slate-500">FLOPs</span>
                  <span className="text-slate-300">{model.flops} GFLOPs</span>
                </div>
                <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-violet-500 rounded-full transition-all duration-500"
                    style={{ width: `${(model.flops / maxFlops) * 100}%` }}
                  />
                </div>
              </div>

              {/* Speed Bar */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-slate-500">GPU Speed</span>
                  <span className="text-slate-300">{model.speed} ms</span>
                </div>
                <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                  {/* Smaller speed is better, so width = (max - val) or just direct ratio */}
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-500"
                    style={{ width: `${(model.speed / maxSpeed) * 100}%` }}
                  />
                </div>
              </div>

              {/* accuracy (mAP) */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-slate-500">mAP50-95</span>
                  <span className="text-slate-300">{model.map}%</span>
                </div>
                <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${(model.map / maxMap) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="border-t border-white/5 pt-2.5 mt-auto">
              <span className="text-[10px] text-slate-500 font-mono">Ideal For:</span>
              <p className="text-[11px] text-emerald-400 font-medium">{model.idealFor}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
