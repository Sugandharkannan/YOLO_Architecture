import React, { useState, useEffect, useRef } from 'react';
import { Upload, Cpu, Sliders, Play, Settings, AlertTriangle, Layers, Info, Check, Activity, RefreshCw } from 'lucide-react';
import { ArchitectureGraph } from './components/ArchitectureGraph';
import { FeatureMapGrid } from './components/FeatureMapGrid';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { ModelComparer } from './components/ModelComparer';

interface Prediction {
  bbox: number[];
  confidence: number;
  class_id: number;
  class_name: string;
}

interface ImageMetadata {
  size_bytes: number;
  resolution: string;
  channels: number;
  filename: string;
  image_base64: string;
  predictions: Prediction[];
}

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

export default function App() {
  // Tabs
  const [activeTab, setActiveTab] = useState<'explore' | 'gallery' | 'compare'>('explore');

  // App State
  const [modelVersion, setModelVersion] = useState<string>('YOLOv8');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [meta, setMeta] = useState<ImageMetadata | null>(null);
  const [layers, setLayers] = useState<LayerInfo[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<string>('Layer 2 (C2f)');
  
  // All Layers Heatmap State
  const [allLayerHeatmaps, setAllLayerHeatmaps] = useState<any[]>([]);
  const [allLayersLoading, setAllLayersLoading] = useState<boolean>(false);
  
  // Feature Map & CAM State
  const [featureMaps, setFeatureMaps] = useState<any>(null);
  const [featureMapsLoading, setFeatureMapsLoading] = useState<boolean>(false);
  const [selectedBbox, setSelectedBbox] = useState<Prediction | null>(null);
  const [targetedCam, setTargetedCam] = useState<string | null>(null);
  const [camLoading, setCamLoading] = useState<boolean>(false);
  const [camOpacity, setCamOpacity] = useState<number>(0.65);
  const [showCam, setShowCam] = useState<boolean>(false);
  const [bboxExplanation, setBboxExplanation] = useState<any>(null);

  // Hyperparameters Playground
  const [lr, setLr] = useState<number>(0.01);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [epochs, setEpochs] = useState<number>(50);
  const [momentum, setMomentum] = useState<number>(0.937);
  const [weightDecay, setWeightDecay] = useState<number>(0.0005);
  
  const [boxLossW, setBoxLossW] = useState<number>(7.5);
  const [clsLossW, setClsLossW] = useState<number>(0.5);
  const [dflLossW, setDflLossW] = useState<number>(1.5);
  
  const [mosaic, setMosaic] = useState<number>(1.0);
  const [mixup, setMixup] = useState<number>(0.0);
  const [hsvH, setHsvH] = useState<number>(0.015);
  const [rotation, setRotation] = useState<number>(0.0);
  const [scaling, setScaling] = useState<number>(0.5);
  const [flipping, setFlipping] = useState<number>(0.5);
  
  const [confThres, setConfThres] = useState<number>(0.25);
  const [iouThres, setIouThres] = useState<number>(0.7);
  const [nmsThres, setNmsThres] = useState<number>(0.45);

  // Simulator Data
  const [metricsBefore, setMetricsBefore] = useState<any>(null);
  const [metricsAfter, setMetricsAfter] = useState<any>(null);
  const [curves, setCurves] = useState<any[]>([]);
  const [gradientFlow, setGradientFlow] = useState<any>(null);
  const [weightUpdateData, setWeightUpdateData] = useState<any>(null);
  const [simLoading, setSimLoading] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch layer structures and default simulator parameters on load
  useEffect(() => {
    fetchLayerDetails();
    runSimulator(true); // Populate baseline metrics
  }, [modelVersion]);

  // Fetch feature maps whenever selected layer or image meta changes
  useEffect(() => {
    if (meta && selectedLayer) {
      fetchFeatureMaps(selectedLayer);
    }
  }, [selectedLayer, meta]);

  // Update Weight matrices when layer changes
  useEffect(() => {
    if (selectedLayer) {
      fetchWeightUpdates(selectedLayer);
    }
  }, [selectedLayer]);

  const fetchLayerDetails = async () => {
    try {
      const res = await fetch(`/api/layer-details?model_version=${modelVersion}`);
      const data = await res.json();
      setLayers(data);
    } catch (e) {
      console.error("Failed to load architecture layers", e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_version', modelVersion);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setMeta(data);
      setSelectedBbox(null);
      setTargetedCam(null);
      setShowCam(false);
      setBboxExplanation(null);
      setAllLayerHeatmaps([]);
    } catch (e) {
      console.error("Upload failed", e);
    } finally {
      setIsUploading(false);
    }
  };

  const fetchFeatureMaps = async (layerName: string) => {
    setFeatureMapsLoading(true);
    try {
      const res = await fetch(`/api/feature-maps?layer_name=${encodeURIComponent(layerName)}`);
      const data = await res.json();
      setFeatureMaps(data);
    } catch (e) {
      console.error("Failed to get feature maps", e);
    } finally {
      setFeatureMapsLoading(false);
    }
  };

  const fetchWeightUpdates = async (layerName: string) => {
    try {
      const res = await fetch(`/api/weight-updates?layer_name=${encodeURIComponent(layerName)}`);
      const data = await res.json();
      setWeightUpdateData(data);
    } catch (e) {
      console.error("Failed to load weight updates", e);
    }
  };

  const handleBboxClick = async (box: Prediction) => {
    setSelectedBbox(box);
    setCamLoading(true);
    setShowCam(true);
    
    try {
      // 1. Fetch targeted Grad-CAM
      const bboxJson = JSON.stringify(box.bbox);
      const camRes = await fetch(`/api/cam?layer_name=${encodeURIComponent(selectedLayer)}&bbox_json=${encodeURIComponent(bboxJson)}`);
      const camData = await camRes.json();
      setTargetedCam(camData.cam_base64);

      // 2. Fetch explanation
      const expRes = await fetch(`/api/explain?class_name=${encodeURIComponent(box.class_name)}&confidence=${box.confidence}`);
      const expData = await expRes.json();
      setBboxExplanation(expData);
    } catch (e) {
      console.error("Failed to fetch explainability payload", e);
    } finally {
      setCamLoading(false);
    }
  };

  const runSimulator = async (isBaseline: boolean = false) => {
    setSimLoading(true);
    const params = {
      lr,
      batch_size: batchSize,
      epochs,
      momentum,
      weight_decay: weightDecay,
      box_loss_w: boxLossW,
      cls_loss_w: clsLossW,
      dfl_loss_w: dflLossW,
      mosaic,
      mixup,
      hsv_h: hsvH,
      rotation,
      scaling,
      flipping,
      conf_thres: confThres,
      iou_thres: iouThres,
      nms_thres: nmsThres
    };

    try {
      // Fetch training metrics & curves
      const resSim = await fetch('/api/simulate-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const simData = await resSim.json();

      setCurves(simData.curves);
      if (isBaseline) {
        setMetricsBefore(simData.metrics);
        setMetricsAfter(simData.metrics);
      } else {
        setMetricsAfter(simData.metrics);
      }

      // Fetch gradient flow
      const resGrad = await fetch('/api/gradient-flow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const gradData = await resGrad.json();
      setGradientFlow(gradData);

    } catch (e) {
      console.error("Simulator execution failed", e);
    } finally {
      setSimLoading(false);
    }
  };

  const fetchAllLayerHeatmaps = async () => {
    if (!meta) return;
    setAllLayersLoading(true);
    try {
      const res = await fetch('/api/all-layer-heatmaps');
      if (res.ok) {
        const data = await res.json();
        setAllLayerHeatmaps(data);
      }
    } catch (e) {
      console.error("Failed to load all layer heatmaps", e);
    } finally {
      setAllLayersLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'gallery' && meta && allLayerHeatmaps.length === 0) {
      fetchAllLayerHeatmaps();
    }
  }, [activeTab, meta]);

  // Helper to draw loss SVG graphs
  const renderLossGraph = () => {
    if (curves.length === 0) return null;
    const padding = 10;
    const w = 240;
    const h = 80;
    
    const validCurves = curves.filter(c => c.train_loss !== "NaN" && c.val_loss !== "NaN");
    if (validCurves.length === 0) return (
      <div className="flex items-center justify-center h-[80px] bg-red-950/20 text-red-400 text-xs font-mono rounded">
        NaN / Exploding Gradients
      </div>
    );

    const maxLoss = Math.max(...validCurves.map(c => Math.max(c.train_loss, c.val_loss)));
    const minLoss = Math.min(...validCurves.map(c => Math.min(c.train_loss, c.val_loss)));
    const range = (maxLoss - minLoss) || 1;

    const getX = (index: number) => padding + (index / (curves.length - 1)) * (w - 2 * padding);
    const getY = (val: number) => h - padding - ((val - minLoss) / range) * (h - 2 * padding);

    const trainPath = validCurves.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(c.train_loss)}`).join(' ');
    const valPath = validCurves.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(c.val_loss)}`).join(' ');

    return (
      <svg className="w-full h-full" viewBox={`0 0 ${w} ${h}`}>
        {/* Grids */}
        <line x1={padding} y1={h - padding} x2={w - padding} y2={h - padding} stroke="#334155" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={h - padding} stroke="#334155" strokeWidth="1" />
        {/* Paths */}
        <path d={trainPath} fill="none" stroke="#6366f1" strokeWidth="1.5" />
        <path d={valPath} fill="none" stroke="#a855f7" strokeWidth="1.5" strokeDasharray="3,2" />
      </svg>
    );
  };

  // Helper to draw accuracy SVG graphs
  const renderAccuracyGraph = () => {
    if (curves.length === 0) return null;
    const padding = 10;
    const w = 240;
    const h = 80;

    const getX = (index: number) => padding + (index / (curves.length - 1)) * (w - 2 * padding);
    const getY = (val: number) => h - padding - (val * (h - 2 * padding));

    const map50Path = curves.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(c.mAP50)}`).join(' ');
    const map95Path = curves.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(c.mAP50_95)}`).join(' ');

    return (
      <svg className="w-full h-full" viewBox={`0 0 ${w} ${h}`}>
        {/* Grids */}
        <line x1={padding} y1={h - padding} x2={w - padding} y2={h - padding} stroke="#334155" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={h - padding} stroke="#334155" strokeWidth="1" />
        {/* Paths */}
        <path d={map50Path} fill="none" stroke="#10b981" strokeWidth="1.5" />
        <path d={map95Path} fill="none" stroke="#06b6d4" strokeWidth="1.5" strokeDasharray="3,2" />
      </svg>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased">
      {/* Premium Navigation Header */}
      <header className="border-b border-white/5 bg-slate-950/80 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2.5 rounded-xl shadow-lg shadow-indigo-600/30 flex items-center justify-center">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              YOLO Vision Explorer
              <span className="text-[10px] uppercase font-mono tracking-widest bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded">
                v1.0.0
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Interactive deep learning debugging & layers explainability dashboard
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-3">
          <nav className="flex bg-slate-900 border border-white/5 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab('explore')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all duration-200 ${
                activeTab === 'explore'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Layer Explorer
            </button>
            <button
              onClick={() => setActiveTab('gallery')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all duration-200 ${
                activeTab === 'gallery'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Layer Flow Gallery
            </button>
            <button
              onClick={() => setActiveTab('compare')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all duration-200 ${
                activeTab === 'compare'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Compare Models
            </button>
          </nav>
        </div>
      </header>

      {/* Tab Contents */}
      <main className="flex-1 p-6 max-w-8xl mx-auto w-full">
        {activeTab === 'compare' ? (
          <div className="glass-panel rounded-2xl p-6 shadow-xl">
            <ModelComparer />
          </div>
        ) : activeTab === 'gallery' ? (
          <div className="glass-panel rounded-2xl p-6 shadow-xl space-y-6">
            <div>
              <h3 className="font-semibold text-lg text-slate-100">
                Layer-by-Layer Flow Gallery
              </h3>
              <p className="text-xs text-slate-400">
                Visualize the progressive feature transformation of the image as it passes through the network. Click any card to inspect in detail.
              </p>
            </div>

            {!meta ? (
              <div className="border border-dashed border-slate-800 rounded-xl p-12 flex flex-col items-center justify-center text-center space-y-2">
                <Info className="w-8 h-8 text-slate-600 animate-pulse" />
                <p className="text-sm text-slate-400">Upload an image in the "Layer Explorer" tab to generate the progression gallery.</p>
              </div>
            ) : allLayersLoading ? (
              <div className="flex flex-col items-center justify-center py-20 space-y-3">
                <div className="w-10 h-10 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
                <p className="text-sm text-slate-400">Extracting features and rendering heatmaps for all hidden layers...</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                {/* Input Image Card */}
                <div className="border border-white/10 bg-slate-950/80 rounded-xl p-3 flex flex-col space-y-2">
                  <div className="flex justify-between items-center text-[10px] font-mono text-emerald-400 font-medium">
                    <span>Input Image</span>
                  </div>
                  <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
                    <img
                      src={`data:image/jpeg;base64,${meta.image_base64}`}
                      alt="Input"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="text-[10px] text-slate-500 text-center font-mono">
                    {meta.resolution}
                  </div>
                </div>

                {/* All Hidden Layers Cards */}
                {allLayerHeatmaps.map((layer) => {
                  const matchedLayer = layers.find(l => l.name === layer.layer_name);
                  const shape = matchedLayer ? matchedLayer.output_shape : "Dynamic";
                  
                  return (
                    <button
                      key={layer.layer_name}
                      onClick={() => {
                        setSelectedLayer(layer.layer_name);
                        setActiveTab('explore');
                      }}
                      className="border border-white/5 hover:border-indigo-500 bg-slate-900/50 hover:bg-slate-900 rounded-xl p-3 flex flex-col space-y-2 text-left group transition-all duration-200 hover:scale-[1.02]"
                    >
                      <div className="flex justify-between items-center text-[10px] font-mono">
                        <span className="text-indigo-400 truncate max-w-[80%]" title={layer.layer_name}>
                          {layer.layer_name.replace("Layer ", "L")}
                        </span>
                        <span className="text-[8px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-sans font-medium uppercase">
                          {layer.layer_type}
                        </span>
                      </div>
                      
                      <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center relative">
                        <img
                          src={`data:image/jpeg;base64,${layer.image_base64}`}
                          alt={layer.layer_name}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                      </div>
                      
                      <div className="text-[10px] text-slate-400 font-mono text-center truncate">
                        {shape}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
            
            {/* COLUMN 1: Model & Architecture List (Left) */}
            <div className="xl:col-span-3 space-y-6">
              {/* Model Upload and Version */}
              <div className="glass-panel rounded-2xl p-4 flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Select YOLO Model
                  </label>
                  <select
                    value={modelVersion}
                    onChange={(e) => setModelVersion(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2 px-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors"
                  >
                    <option value="YOLOv8">YOLOv8 (Nano)</option>
                    <option value="YOLOv9">YOLOv9 (Compact)</option>
                    <option value="YOLOv10">YOLOv10 (Nano)</option>
                    <option value="YOLOv11">YOLOv11 (Nano)</option>
                    <option value="Custom YOLO model">Custom YOLO model</option>
                  </select>
                </div>

                {/* Upload Area */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Input Image
                  </label>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    className="hidden"
                    accept="image/*"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="w-full border border-dashed border-slate-700 hover:border-slate-600 bg-slate-950/50 rounded-xl p-6 flex flex-col items-center justify-center text-center gap-2 group transition-all duration-200"
                  >
                    {isUploading ? (
                      <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
                    ) : (
                      <Upload className="w-6 h-6 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                    )}
                    <span className="text-xs font-medium text-slate-300">
                      {isUploading ? 'Extracting features...' : 'Upload Image'}
                    </span>
                    <span className="text-[10px] text-slate-500">Supports PNG, JPG, JPEG</span>
                  </button>
                </div>

                {/* File Metadata */}
                {meta && (
                  <div className="bg-slate-950/60 rounded-xl p-3 border border-white/5 space-y-1.5 text-xs font-mono text-slate-400">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Resolution:</span>
                      <span className="text-slate-300 font-medium">{meta.resolution}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">File Size:</span>
                      <span className="text-slate-300 font-medium">{(meta.size_bytes / 1024).toFixed(1)} KB</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Channels:</span>
                      <span className="text-slate-300 font-medium">{meta.channels} (RGB)</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Architecture Explorer List */}
              <div className="glass-panel rounded-2xl p-4">
                <ArchitectureGraph
                  layers={layers}
                  selectedLayer={selectedLayer}
                  onSelectLayer={setSelectedLayer}
                />
              </div>
            </div>

            {/* COLUMN 2: Main Image Overlay & Feature Maps (Middle) */}
            <div className="xl:col-span-6 space-y-6">
              
              {/* Original Image Box with Bounding Box Overlay & Grad-CAM */}
              <div className="glass-panel rounded-2xl p-4 flex flex-col space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-lg text-slate-100">
                      Prediction Visualizer
                    </h3>
                    <p className="text-xs text-slate-400">
                      {meta ? 'Click a bounding box to explain predictions and project Grad-CAM' : 'Upload an image to start inference'}
                    </p>
                  </div>

                  {targetedCam && (
                    <div className="flex items-center gap-4">
                      {/* Grad-CAM toggler */}
                      <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={showCam}
                          onChange={(e) => setShowCam(e.target.checked)}
                          className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 bg-slate-900"
                        />
                        Grad-CAM Overlay
                      </label>
                      {showCam && (
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-slate-500">Opacity:</span>
                          <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={camOpacity}
                            onChange={(e) => setCamOpacity(parseFloat(e.target.value))}
                            className="w-16 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Interactive Canvas container */}
                <div className="relative border border-white/5 bg-slate-950 rounded-xl overflow-hidden min-h-[350px] flex items-center justify-center">
                  {!meta ? (
                    <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
                      <Cpu className="w-10 h-10 mb-2 stroke-1" />
                      <p className="text-sm">Awaiting Image Upload</p>
                    </div>
                  ) : (
                    <div className="relative max-w-full max-h-[500px]">
                      {/* Base Image */}
                      <img
                        src={`data:image/jpeg;base64,${meta.image_base64}`}
                        alt="Uploaded Inference Input"
                        className="max-w-full max-h-[500px] object-contain select-none"
                      />

                      {/* Grad-CAM Heatmap overlay */}
                      {showCam && targetedCam && (
                        <img
                          src={`data:image/jpeg;base64,${targetedCam}`}
                          alt="Grad-CAM targeted heatmap"
                          className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300"
                          style={{ opacity: camOpacity, mixBlendMode: 'screen' }}
                        />
                      )}

                      {/* Render bounding box borders */}
                      {meta.predictions.map((pred, i) => {
                        // We map relative xyxy coordinates to style percentages.
                        // However, to keep it simple, since the aspect ratio is mapped,
                        // we can overlay transparent clickable absolute boxes.
                        // We'll compute bounding box rectangles by checking coordinates
                        const isSelected = selectedBbox?.class_id === pred.class_id;
                        
                        // Fallback coordinates rendering assuming 640 scale
                        const [x1, y1, x2, y2] = pred.bbox;
                        // Let's find percentage coords relative to 640 standard YOLO scale
                        const left = `${(x1 / 640) * 100}%`;
                        const top = `${(y1 / 640) * 100}%`;
                        const width = `${((x2 - x1) / 640) * 100}%`;
                        const height = `${((y2 - y1) / 640) * 100}%`;

                        return (
                          <div
                            key={i}
                            onClick={() => handleBboxClick(pred)}
                            className={`absolute border-2 cursor-pointer group transition-all duration-200 ${
                              isSelected
                                ? 'border-indigo-400 bg-indigo-500/10 shadow-lg'
                                : 'border-emerald-500 hover:border-indigo-500 bg-transparent hover:bg-white/5'
                            }`}
                            style={{
                              left,
                              top,
                              width,
                              height,
                            }}
                          >
                            {/* Class tag */}
                            <span className="absolute -top-5 left-0 bg-emerald-600 text-white font-mono text-[9px] px-1.5 py-0.5 rounded flex items-center gap-1 shadow-sm select-none">
                              {pred.class_name} {(pred.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {camLoading && (
                    <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                      <div className="w-6 h-6 border-2 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
                    </div>
                  )}
                </div>

                {/* Explanation text for prediction click */}
                {bboxExplanation && (
                  <div className="border border-indigo-500/20 bg-indigo-950/20 rounded-xl p-3.5 space-y-2 animate-fadeIn">
                    <div className="flex items-center gap-2">
                      <Info className="w-4 h-4 text-indigo-400" />
                      <span className="font-semibold text-xs text-indigo-300 uppercase tracking-wider">
                        Explain Prediction: {bboxExplanation.class_name} ({bboxExplanation.confidence})
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {bboxExplanation.summary}
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider">
                          Key Features Activated:
                        </span>
                        <ul className="list-disc pl-4 text-xs text-slate-400 space-y-0.5 mt-1">
                          {bboxExplanation.reasons.map((r: string, idx: number) => (
                            <li key={idx}>{r}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider">
                          Contributing Layers:
                        </span>
                        <ul className="list-disc pl-4 text-[11px] font-mono text-indigo-400 space-y-0.5 mt-1">
                          {bboxExplanation.contributing_layers.map((l: string, idx: number) => (
                            <li key={idx}>{l}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Feature Maps Explorer */}
              <div className="glass-panel rounded-2xl p-4">
                <FeatureMapGrid
                  layerName={selectedLayer}
                  channelsCount={featureMaps?.channels_count || 0}
                  eigenCamBase64={featureMaps?.eigen_cam_base64 || null}
                  topChannels={featureMaps?.top_channels || []}
                  loading={featureMapsLoading}
                  explanation={featureMaps?.explanation}
                />
              </div>
            </div>

            {/* COLUMN 3: Training Playground & Metrics (Right) */}
            <div className="xl:col-span-3 space-y-6">
              
              {/* Hyperparameter Playground */}
              <div className="glass-panel rounded-2xl p-4 space-y-4">
                <div className="flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-indigo-400" />
                  <h3 className="font-semibold text-sm uppercase tracking-wider text-slate-200">
                    Hyperparameter Playground
                  </h3>
                </div>

                <div className="space-y-3.5 max-h-[300px] overflow-y-auto pr-1">
                  {/* Learning Rate */}
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Learning Rate (LR)</span>
                      <span className="text-indigo-400 font-mono font-medium">{lr}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0001"
                      max="0.1"
                      step="0.001"
                      value={lr}
                      onChange={(e) => setLr(parseFloat(e.target.value))}
                      className="h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                  </div>

                  {/* Batch Size */}
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Batch Size</span>
                      <span className="text-indigo-400 font-mono font-medium">{batchSize}</span>
                    </div>
                    <select
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                      className="bg-slate-950 border border-slate-850 rounded py-1 px-2 text-xs font-mono"
                    >
                      <option value={8}>8</option>
                      <option value={16}>16</option>
                      <option value={32}>32</option>
                      <option value={64}>64</option>
                    </select>
                  </div>

                  {/* Loss Weights */}
                  <div className="border-t border-white/5 pt-2.5 space-y-2">
                    <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">Loss Weights</span>
                    
                    <div className="grid grid-cols-3 gap-2">
                      <div className="flex flex-col gap-1 text-[11px]">
                        <span className="text-slate-400">Box Loss</span>
                        <input
                          type="number"
                          value={boxLossW}
                          onChange={(e) => setBoxLossW(parseFloat(e.target.value))}
                          className="bg-slate-950 border border-slate-850 rounded p-1 text-center font-mono"
                        />
                      </div>
                      <div className="flex flex-col gap-1 text-[11px]">
                        <span className="text-slate-400">Class Loss</span>
                        <input
                          type="number"
                          value={clsLossW}
                          onChange={(e) => setClsLossW(parseFloat(e.target.value))}
                          className="bg-slate-950 border border-slate-850 rounded p-1 text-center font-mono"
                        />
                      </div>
                      <div className="flex flex-col gap-1 text-[11px]">
                        <span className="text-slate-400">DFL Loss</span>
                        <input
                          type="number"
                          value={dflLossW}
                          onChange={(e) => setDflLossW(parseFloat(e.target.value))}
                          className="bg-slate-950 border border-slate-850 rounded p-1 text-center font-mono"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Augmentations */}
                  <div className="border-t border-white/5 pt-2.5 space-y-2.5">
                    <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">Augmentations</span>
                    
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Mosaic Probability</span>
                        <span className="text-slate-300 font-mono">{mosaic}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={mosaic}
                        onChange={(e) => setMosaic(parseFloat(e.target.value))}
                        className="h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Mixup Probability</span>
                        <span className="text-slate-300 font-mono">{mixup}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={mixup}
                        onChange={(e) => setMixup(parseFloat(e.target.value))}
                        className="h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>
                  </div>

                  {/* Conf & IoU Limits */}
                  <div className="border-t border-white/5 pt-2.5 space-y-2.5">
                    <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">Inference Thresholds</span>
                    
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">Confidence Threshold</span>
                        <span className="text-slate-300 font-mono">{confThres}</span>
                      </div>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={confThres}
                        onChange={(e) => setConfThres(parseFloat(e.target.value))}
                        className="h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">IoU Threshold</span>
                        <span className="text-slate-300 font-mono">{iouThres}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={iouThres}
                        onChange={(e) => setIouThres(parseFloat(e.target.value))}
                        className="h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => runSimulator(false)}
                  disabled={simLoading}
                  className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800/50 rounded-lg text-xs font-semibold text-white tracking-wide transition-all shadow-md flex items-center justify-center gap-1.5"
                >
                  {simLoading ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-white" />
                  )}
                  Apply & Run Simulation
                </button>
              </div>

              {/* Pipeline Step Visualizer */}
              <PipelineVisualizer />

              {/* Live Impact Analysis Curves */}
              <div className="glass-panel rounded-2xl p-4 space-y-4">
                <h3 className="font-semibold text-sm uppercase tracking-wider text-slate-200">
                  Live Impact Analysis
                </h3>
                
                {/* Before vs After stats */}
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="bg-slate-950/60 rounded-xl p-2.5 border border-white/5">
                    <span className="text-[10px] text-slate-500 block">Baseline mAP</span>
                    <span className="text-base font-bold text-slate-400 font-mono">
                      {metricsBefore ? `${(metricsBefore.mAP50 * 100).toFixed(0)}%` : '--'}
                    </span>
                  </div>
                  <div className="bg-indigo-950/30 rounded-xl p-2.5 border border-indigo-500/10">
                    <span className="text-[10px] text-indigo-400 block font-medium">Estimated mAP</span>
                    <span className="text-base font-bold text-indigo-300 font-mono">
                      {metricsAfter ? `${(metricsAfter.mAP50 * 100).toFixed(0)}%` : '--'}
                    </span>
                  </div>
                </div>

                {/* Small Graphs */}
                <div className="space-y-3">
                  {/* Loss Curve */}
                  <div className="flex flex-col gap-1 border border-white/5 bg-slate-950/40 rounded-lg p-2">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-slate-400 font-semibold uppercase tracking-wider">Training Loss Curve</span>
                      <span className="text-indigo-400 font-mono">Train (solid) vs Val (dash)</span>
                    </div>
                    <div className="h-20 flex items-center justify-center">
                      {renderLossGraph()}
                    </div>
                  </div>

                  {/* Accuracy Curve */}
                  <div className="flex flex-col gap-1 border border-white/5 bg-slate-950/40 rounded-lg p-2">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-slate-400 font-semibold uppercase tracking-wider">mAP Accuracy Curve</span>
                      <span className="text-emerald-400 font-mono">mAP50 (solid) vs mAP50-95 (dash)</span>
                    </div>
                    <div className="h-20 flex items-center justify-center">
                      {renderAccuracyGraph()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Gradient Flow Visualizer */}
              {gradientFlow && (
                <div className="glass-panel rounded-2xl p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="font-semibold text-sm uppercase tracking-wider text-slate-200">
                      Gradient Flow
                    </h3>
                    
                    {/* Vanishing/Exploding Gradient Warnings */}
                    {gradientFlow.status !== "Normal" && (
                      <span className="text-[9px] bg-red-950 border border-red-500/20 text-red-400 px-2 py-0.5 rounded uppercase font-semibold flex items-center gap-1 font-mono">
                        <AlertTriangle className="w-2.5 h-2.5" />
                        {gradientFlow.status}
                      </span>
                    )}
                  </div>
                  
                  <p className="text-[10px] text-slate-400 leading-relaxed leading-tight">
                    {gradientFlow.description}
                  </p>

                  {/* SVG Bar Chart for gradient magnitudes */}
                  <div className="h-24 bg-slate-950/50 rounded-lg p-2 flex items-end justify-between border border-white/5 gap-1.5">
                    {gradientFlow.magnitudes.map((mag: number, idx: number) => {
                      // Normalise the magnitude visual heights
                      const maxVal = Math.max(...gradientFlow.magnitudes) || 1;
                      const heightPercent = `${Math.max(4, (mag / maxVal) * 90)}%`;
                      const layerShortName = gradientFlow.layers[idx].replace("Layer ", "").slice(0, 7);
                      
                      return (
                        <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group relative cursor-pointer">
                          {/* Tooltip */}
                          <div className="absolute bottom-full mb-1 bg-slate-900 border border-slate-700 text-white text-[9px] font-mono p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity z-20 pointer-events-none">
                            {mag}
                          </div>
                          
                          <div
                            className={`w-full rounded-t-sm transition-all duration-300 ${
                              gradientFlow.status === 'Exploding'
                                ? 'bg-red-500 shadow-md shadow-red-500/20'
                                : gradientFlow.status === 'Vanishing'
                                ? 'bg-amber-600'
                                : 'bg-indigo-500 shadow-md shadow-indigo-500/20'
                            }`}
                            style={{ height: heightPercent }}
                          />
                          <span className="text-[8px] text-slate-500 font-mono mt-1 select-none">
                            L{idx}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Weights Update Visualizer */}
              {weightUpdateData && (
                <div className="glass-panel rounded-2xl p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="font-semibold text-sm uppercase tracking-wider text-slate-200">
                      Weight Matrix Update
                    </h3>
                    <span className="text-[10px] bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-mono px-2 py-0.5 rounded">
                      +{weightUpdateData.avg_pct_update}% delta
                    </span>
                  </div>

                  <p className="text-[10px] text-slate-400">
                    Comparing kernel weights distribution of <span className="font-mono text-indigo-400">{selectedLayer}</span> pre/post epoch training:
                  </p>

                  <div className="grid grid-cols-2 gap-3.5">
                    {/* Weight Matrix A (Pre) */}
                    <div className="flex flex-col gap-1 border border-white/5 bg-slate-950/40 rounded-lg p-2 text-center">
                      <span className="text-[9px] uppercase tracking-wider text-slate-500">Matrix A (Pre-train)</span>
                      <div className="grid grid-cols-8 gap-0.5 mt-1 border border-slate-900 p-0.5 bg-black rounded">
                        {weightUpdateData.matrix_a.flatMap((row: number[]) => row).map((val: number, i: number) => {
                          // Map value to colors
                          // Val is normally around -0.5 to +0.5
                          const absVal = Math.min(1.0, Math.abs(val) * 2.0);
                          const bg = val > 0 
                            ? `rgba(99, 102, 241, ${absVal})` // Blue/indigo
                            : `rgba(239, 68, 68, ${absVal})`; // Red
                          return (
                            <div
                              key={i}
                              className="aspect-square rounded-sm"
                              style={{ backgroundColor: bg }}
                              title={`Val: ${val.toFixed(4)}`}
                            />
                          );
                        })}
                      </div>
                    </div>

                    {/* Weight Matrix B (Post) */}
                    <div className="flex flex-col gap-1 border border-white/5 bg-slate-950/40 rounded-lg p-2 text-center">
                      <span className="text-[9px] uppercase tracking-wider text-slate-500">Matrix B (Post-train)</span>
                      <div className="grid grid-cols-8 gap-0.5 mt-1 border border-slate-900 p-0.5 bg-black rounded">
                        {weightUpdateData.matrix_b.flatMap((row: number[]) => row).map((val: number, i: number) => {
                          const absVal = Math.min(1.0, Math.abs(val) * 2.0);
                          const bg = val > 0 
                            ? `rgba(99, 102, 241, ${absVal})` 
                            : `rgba(239, 68, 68, ${absVal})`;
                          return (
                            <div
                              key={i}
                              className="aspect-square rounded-sm"
                              style={{ backgroundColor: bg }}
                              title={`Val: ${val.toFixed(4)}`}
                            />
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}

            </div>

          </div>
        )}
      </main>

      {/* Styled Footer */}
      <footer className="border-t border-white/5 bg-slate-950 py-6 mt-12 text-center text-xs text-slate-500 font-mono">
        <p>YOLO Vision Explorer © 2026. Made with React, Next.js and PyTorch/FastAPI.</p>
      </footer>
    </div>
  );
}
