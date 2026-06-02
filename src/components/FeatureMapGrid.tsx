import React, { useState } from 'react';
import { ZoomIn, Columns, Maximize2, X, AlertCircle } from 'lucide-react';

interface ChannelData {
  channel_id: number;
  description: string;
  image_base64: string;
}

interface LayerExplanation {
  role: string;
  summary: string;
  features_learned: string[];
  active_regions: string;
  important_channels: string;
}

interface FeatureMapGridProps {
  layerName: string;
  channelsCount: number;
  eigenCamBase64: string | null;
  topChannels: ChannelData[];
  loading: boolean;
  explanation?: LayerExplanation;
}

export const FeatureMapGrid: React.FC<FeatureMapGridProps> = ({
  layerName,
  channelsCount,
  eigenCamBase64,
  topChannels,
  loading,
  explanation,
}) => {
  const [zoomedImage, setZoomedImage] = useState<{ src: string; title: string; desc: string } | null>(null);
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([]);
  const [compareMode, setCompareMode] = useState(false);

  const toggleCompare = (channelId: number) => {
    if (selectedForCompare.includes(channelId)) {
      setSelectedForCompare(selectedForCompare.filter((id) => id !== channelId));
    } else {
      if (selectedForCompare.length < 2) {
        setSelectedForCompare([...selectedForCompare, channelId]);
      } else {
        // Swap out the first one if already 2 selected
        setSelectedForCompare([selectedForCompare[1], channelId]);
      }
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-lg text-slate-100">
            Feature Map Explorer
          </h3>
          <p className="text-xs text-slate-400">
            Showing activations for <span className="text-indigo-400 font-medium font-mono">{layerName}</span> ({channelsCount} channels)
          </p>
        </div>
        
        {topChannels.length > 0 && (
          <button
            onClick={() => {
              setCompareMode(!compareMode);
              setSelectedForCompare([]);
            }}
            className={`px-3 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-all duration-200 ${
              compareMode
                ? 'bg-indigo-600 border-indigo-500 text-white shadow-indigo-600/30 shadow-md'
                : 'bg-slate-800/80 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            {compareMode ? 'Exit Compare' : 'Compare Channels'}
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center py-12 space-y-3">
          <div className="w-10 h-10 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
          <p className="text-sm text-slate-400">Extracting layer activation tensors...</p>
        </div>
      ) : topChannels.length === 0 ? (
        <div className="flex-1 border border-dashed border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center text-center space-y-2">
          <AlertCircle className="w-8 h-8 text-slate-600" />
          <p className="text-sm text-slate-400">Upload an image and select a layer block to inspect feature maps.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1">
          {/* Saliency & Attention (Eigen-CAM) Map */}
          <div className="lg:col-span-1 flex flex-col border border-white/5 bg-slate-900/40 rounded-xl p-3.5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Layer Heatmap (Eigen-CAM)
              </span>
              <span className="text-[10px] text-emerald-400 font-mono bg-emerald-950/40 border border-emerald-500/10 px-2 py-0.5 rounded">
                Principal Features
              </span>
            </div>
            
            {eigenCamBase64 && (
              <div className="relative group overflow-hidden rounded-lg border border-white/5 bg-black flex-1 flex items-center justify-center min-h-[220px]">
                <img
                  src={`data:image/jpeg;base64,${eigenCamBase64}`}
                  alt="Eigen-CAM Activation Map"
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                  <button
                    onClick={() =>
                      setZoomedImage({
                        src: `data:image/jpeg;base64,${eigenCamBase64}`,
                        title: 'Eigen-CAM Heatmap',
                        desc: `Principal spatial activations overlay for ${layerName}. Highlights which pixels contributed the most features.`,
                      })
                    }
                    className="p-2 bg-indigo-600 rounded-full hover:scale-110 transition-transform text-white shadow-lg"
                  >
                    <ZoomIn className="w-5 h-5" />
                  </button>
                </div>
              </div>
            )}
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Eigen-CAM visualizes the principal components of the activation maps. Bright yellow/red regions are where the model forms high-level feature extractions.
            </p>
          </div>

          {/* Top Channels Grid */}
          <div className="lg:col-span-2 flex flex-col border border-white/5 bg-slate-900/40 rounded-xl p-3.5 space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              {compareMode ? 'Select 2 Channels to Compare' : 'Top Activated Channel Filters'}
            </span>

            {compareMode ? (
              /* Compare panel */
              <div className="flex-1 flex flex-col space-y-3">
                <div className="grid grid-cols-2 gap-4 flex-1">
                  {selectedForCompare.map((channelId) => {
                    const ch = topChannels.find((c) => c.channel_id === channelId);
                    if (!ch) return null;
                    return (
                      <div key={channelId} className="border border-white/5 bg-black/30 rounded-lg p-2 flex flex-col space-y-2">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className="text-indigo-400 font-medium">Channel {channelId}</span>
                          <button
                            onClick={() => toggleCompare(channelId)}
                            className="text-slate-400 hover:text-slate-200"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <div className="relative flex-1 flex items-center justify-center overflow-hidden rounded bg-black">
                          <img
                            src={`data:image/jpeg;base64,${ch.image_base64}`}
                            alt={`Channel ${channelId}`}
                            className="w-full max-h-[200px] object-cover"
                          />
                        </div>
                        <p className="text-[10px] text-slate-400 italic text-center font-sans">{ch.description}</p>
                      </div>
                    );
                  })}
                  {selectedForCompare.length < 2 && (
                    <div className="border border-dashed border-slate-800 rounded-lg flex flex-col items-center justify-center text-center p-4 min-h-[180px]">
                      <Columns className="w-6 h-6 text-slate-700 mb-1" />
                      <p className="text-[11px] text-slate-500">
                        {selectedForCompare.length === 0
                          ? 'Select two channels below to place them side by side'
                          : 'Select one more channel below'}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            {/* Channels List */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 overflow-y-auto max-h-[300px] pr-1">
              {topChannels.map((ch) => {
                const isSelected = selectedForCompare.includes(ch.channel_id);
                return (
                  <div
                    key={ch.channel_id}
                    className={`group relative border rounded-lg p-2 flex flex-col gap-1.5 transition-all duration-200 ${
                      compareMode
                        ? isSelected
                          ? 'border-indigo-500 bg-indigo-950/20'
                          : 'border-slate-800 hover:border-slate-700 bg-slate-900/50 cursor-pointer'
                        : 'border-white/5 bg-slate-900/50'
                    }`}
                    onClick={() => compareMode && toggleCompare(ch.channel_id)}
                  >
                    <div className="flex justify-between items-center text-[10px] font-mono">
                      <span className="text-slate-400 font-medium">Ch {ch.channel_id}</span>
                      {compareMode && (
                        <div
                          className={`w-3.5 h-3.5 rounded border transition-colors flex items-center justify-center ${
                            isSelected
                              ? 'bg-indigo-500 border-indigo-400 text-white text-[8px]'
                              : 'border-slate-600'
                          }`}
                        >
                          {isSelected && '✓'}
                        </div>
                      )}
                    </div>
                    
                    <div className="relative overflow-hidden rounded bg-black flex items-center justify-center aspect-square">
                      <img
                        src={`data:image/jpeg;base64,${ch.image_base64}`}
                        alt={`Channel ${ch.channel_id}`}
                        className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
                      />
                      {!compareMode && (
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex items-center justify-center">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setZoomedImage({
                                src: `data:image/jpeg;base64,${ch.image_base64}`,
                                title: `Channel ${ch.channel_id} Activation`,
                                desc: ch.description,
                              });
                            }}
                            className="p-1.5 bg-indigo-600 rounded-full hover:scale-115 transition-transform text-white"
                          >
                            <Maximize2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      )}
                    </div>
                    
                    <p className="text-[9px] text-slate-400 truncate tracking-wide" title={ch.description}>
                      {ch.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* AI Layer Explanation Card */}
      {!loading && topChannels.length > 0 && explanation && (
        <div className="border border-indigo-500/20 bg-indigo-950/15 rounded-xl p-4.5 space-y-3 animate-fadeIn">
          <div className="flex items-center justify-between border-b border-white/5 pb-2">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse"></span>
              <span className="font-semibold text-xs text-indigo-300 uppercase tracking-wider font-mono">
                AI Layer Analysis: {explanation.role}
              </span>
            </div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-mono">
              Semantic Weighting Guide
            </span>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            {explanation.summary}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs pt-2">
            {/* Features Learned */}
            <div className="bg-slate-950/40 border border-white/5 rounded-lg p-2.5 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider font-semibold">
                Features Extracted:
              </span>
              <div className="flex flex-wrap gap-1 mt-1">
                {explanation.features_learned.map((feat, idx) => (
                  <span key={idx} className="bg-indigo-500/10 text-indigo-300 text-[10px] px-2 py-0.5 rounded border border-indigo-500/10">
                    {feat}
                  </span>
                ))}
              </div>
            </div>

            {/* Active Regions */}
            <div className="bg-slate-950/40 border border-white/5 rounded-lg p-2.5 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider font-semibold">
                Most Active Spatial Regions:
              </span>
              <p className="text-[11px] text-slate-300 italic mt-1 leading-relaxed">
                {explanation.active_regions}
              </p>
            </div>

            {/* Important Channels */}
            <div className="bg-slate-950/40 border border-white/5 rounded-lg p-2.5 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider font-semibold">
                Important Channels:
              </span>
              <p className="text-[11px] text-indigo-400 font-mono mt-1 font-semibold">
                Channels {explanation.important_channels}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Zoom Modal */}
      {zoomedImage && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl overflow-hidden shadow-2xl flex flex-col">
            <div className="flex justify-between items-center p-4 border-b border-white/5">
              <div>
                <h4 className="font-semibold text-slate-100">{zoomedImage.title}</h4>
                <p className="text-xs text-indigo-400 font-mono">{layerName}</p>
              </div>
              <button
                onClick={() => setZoomedImage(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="p-6 bg-black flex justify-center items-center">
              <img
                src={zoomedImage.src}
                alt="Zoomed Activation Map"
                className="max-w-full max-h-[400px] object-contain rounded"
              />
            </div>
            
            <div className="p-4 border-t border-white/5 bg-slate-900/50">
              <p className="text-xs text-slate-300 leading-relaxed">
                {zoomedImage.desc}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
