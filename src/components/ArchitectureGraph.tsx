import React from 'react';
import { Cpu, ArrowDown, Layers, HelpCircle, HardDrive, Zap } from 'lucide-react';

interface LayerInfo {
  name: string;
  type: string;
  input_shape: string;
  output_shape: string;
  parameters: number;
  flops: string;
  memory: string;
  activation: string;
}

interface ArchitectureGraphProps {
  layers: LayerInfo[];
  selectedLayer: string;
  onSelectLayer: (layerName: string) => void;
}

export const ArchitectureGraph: React.FC<ArchitectureGraphProps> = ({
  layers,
  selectedLayer,
  onSelectLayer,
}) => {
  // Helper to color-code layer types
  const getNodeColor = (type: string, isSelected: boolean) => {
    if (isSelected) return 'border-indigo-500 bg-indigo-950/80 shadow-indigo-500/30 text-indigo-200';
    switch (type.toLowerCase()) {
      case 'input':
        return 'border-emerald-500/50 bg-emerald-950/30 text-emerald-300';
      case 'conv':
        return 'border-blue-500/50 bg-blue-950/30 text-blue-300';
      case 'c2f':
        return 'border-violet-500/50 bg-violet-950/30 text-violet-300';
      case 'sppf':
        return 'border-amber-500/50 bg-amber-950/30 text-amber-300';
      case 'concat':
        return 'border-rose-500/50 bg-rose-950/30 text-rose-300';
      default:
        return 'border-slate-500/50 bg-slate-900/50 text-slate-300';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-3">
        <Layers className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold tracking-wide text-slate-100">
          Architecture Explorer
        </h2>
      </div>

      <p className="text-xs text-slate-400 mb-4">
        Click any node in the network to inspect its features, channels, and activation shapes.
      </p>

      <div className="flex-1 overflow-y-auto pr-1 space-y-3 relative max-h-[500px]">
        {layers.map((layer, index) => {
          const isSelected = selectedLayer === layer.name;
          const isLast = index === layers.length - 1;
          
          return (
            <div key={layer.name} className="flex flex-col items-center">
              {/* Layer Node Card */}
              <button
                onClick={() => onSelectLayer(layer.name)}
                className={`w-full text-left p-3 rounded-lg border glass-panel transition-all duration-300 hover:scale-[1.02] flex flex-col gap-1.5 focus:outline-none ${getNodeColor(
                  layer.type,
                  isSelected
                )} ${isSelected ? 'glow-active border-indigo-400' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    {layer.type} Block
                  </span>
                  <span className="text-[10px] bg-slate-800/80 px-2 py-0.5 rounded text-slate-300 font-mono">
                    #{index}
                  </span>
                </div>
                
                <h3 className="font-medium text-sm text-slate-100 truncate">
                  {layer.name}
                </h3>
                
                {/* Active selection stats */}
                <div className="grid grid-cols-2 gap-x-2 gap-y-1 mt-1 text-[11px] font-mono text-slate-400 border-t border-white/5 pt-1.5">
                  <div>
                    <span className="text-slate-500">In:</span> {layer.input_shape}
                  </div>
                  <div>
                    <span className="text-slate-500">Out:</span> {layer.output_shape}
                  </div>
                  {layer.parameters > 0 && (
                    <div>
                      <span className="text-slate-500">Params:</span> {layer.parameters.toLocaleString()}
                    </div>
                  )}
                  {layer.activation !== "None" && (
                    <div>
                      <span className="text-slate-500">Act:</span> {layer.activation}
                    </div>
                  )}
                </div>
              </button>

              {/* Connecting Flow Line */}
              {!isLast && (
                <div className="flex flex-col items-center my-1.5">
                  <svg className="w-6 h-6 text-slate-600" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M12 0v24"
                      className={`${isSelected ? 'stroke-indigo-400 flow-line' : 'stroke-slate-700'}`}
                      strokeWidth="2.5"
                    />
                    <path
                      d="M8 16l4 4 4-4"
                      className={`${isSelected ? 'stroke-indigo-400' : 'stroke-slate-700'}`}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
