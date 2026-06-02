import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ShieldAlert, Settings2 } from 'lucide-react';

interface Step {
  id: number;
  label: string;
  desc: string;
  type: 'forward' | 'backward' | 'loss' | 'update';
}

const PIPELINE_STEPS: Step[] = [
  { id: 1, label: 'Augmentation', desc: 'Applies Mosaic, Mixup, HSV shifts, and scaling to the training image.', type: 'forward' },
  { id: 2, label: 'Forward Pass', desc: 'Processes the augmented image batch through Backbone, Neck, and Head layers.', type: 'forward' },
  { id: 3, label: 'Feature Extraction', desc: 'Extracts multiscale spatial feature maps (P3, P4, P5 grids).', type: 'forward' },
  { id: 4, label: 'Prediction', desc: 'Generates bounding box coords, object confidences, and class probabilities.', type: 'forward' },
  { id: 5, label: 'Loss Calculation', desc: 'Computes CIoU Box Loss, BCE Classification Loss, and Distribution Focal Loss.', type: 'loss' },
  { id: 6, label: 'Backpropagation', desc: 'Propagates loss gradients backward through PyTorch autograd graph.', type: 'backward' },
  { id: 7, label: 'Weight Update', desc: 'Applies optimizer (SGD/AdamW) updates to layer weight matrices based on momentum.', type: 'update' }
];

export const PipelineVisualizer: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setActiveStep((prev) => (prev + 1) % (PIPELINE_STEPS.length + 1));
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleReset = () => {
    setActiveStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="flex flex-col border border-white/5 bg-slate-900/40 rounded-xl p-4 space-y-4">
      {/* Title */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-sm uppercase tracking-wider text-slate-400">
            Training Pipeline Visualizer
          </h3>
          <p className="text-[11px] text-slate-500">
            Interactive training loop flow & gradient propagation sequence
          </p>
        </div>
        
        {/* Playback controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Visual Pipeline Grid */}
      <div className="flex flex-col space-y-2 relative">
        {PIPELINE_STEPS.map((step, idx) => {
          const isHighlighted = activeStep === step.id;
          const isPassed = activeStep > step.id || activeStep === 0;
          
          return (
            <div key={step.id} className="flex items-stretch gap-3 relative">
              {/* Connector line on left */}
              {idx < PIPELINE_STEPS.length - 1 && (
                <div className="absolute left-3.5 top-7 bottom-0 w-0.5 bg-slate-800 z-0">
                  <div
                    className={`w-full h-full bg-indigo-500 transition-all duration-[2500ms] origin-top ${
                      isPassed && activeStep !== 0 ? 'scale-y-100' : 'scale-y-0'
                    }`}
                  />
                </div>
              )}

              {/* Number Badge */}
              <div
                onClick={() => {
                  setActiveStep(step.id);
                  setIsPlaying(false);
                }}
                className={`w-7 h-7 rounded-full border z-10 flex items-center justify-center font-mono text-xs font-semibold cursor-pointer transition-all duration-300 ${
                  isHighlighted
                    ? 'bg-indigo-600 border-indigo-400 text-white glow-active scale-110'
                    : isPassed && activeStep !== 0
                    ? 'bg-indigo-950/60 border-indigo-500/50 text-indigo-300'
                    : 'bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-700'
                }`}
              >
                {step.id}
              </div>

              {/* Step content */}
              <div
                onClick={() => {
                  setActiveStep(step.id);
                  setIsPlaying(false);
                }}
                className={`flex-1 p-2 rounded-lg border cursor-pointer transition-all duration-300 flex flex-col ${
                  isHighlighted
                    ? 'border-indigo-500/40 bg-indigo-950/30'
                    : 'border-slate-800/80 bg-slate-950/30 hover:border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`font-semibold text-xs transition-colors ${
                      isHighlighted ? 'text-indigo-300' : 'text-slate-300'
                    }`}
                  >
                    {step.label}
                  </span>
                  
                  {/* Step Action Tag */}
                  <span
                    className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono font-medium ${
                      step.type === 'forward'
                        ? 'bg-blue-950/40 border border-blue-500/10 text-blue-400'
                        : step.type === 'backward'
                        ? 'bg-rose-950/40 border border-rose-500/10 text-rose-400'
                        : step.type === 'loss'
                        ? 'bg-amber-950/40 border border-amber-500/10 text-amber-400'
                        : 'bg-emerald-950/40 border border-emerald-500/10 text-emerald-400'
                    }`}
                  >
                    {step.type}
                  </span>
                </div>
                
                {isHighlighted && (
                  <p className="text-[10px] text-slate-400 mt-1 leading-relaxed animate-fadeIn">
                    {step.desc}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Epoch animation box at the bottom */}
      <div className="border border-white/5 bg-slate-950/50 rounded-lg p-2.5 flex items-center justify-between text-xs">
        <span className="text-slate-400">Pipeline Status:</span>
        <span className="font-mono text-indigo-400 flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full bg-indigo-500 ${isPlaying ? 'animate-ping' : ''}`}></span>
          {activeStep === 0
            ? 'Idle / Awaiting Batch'
            : `Executing Step ${activeStep}/7: ${PIPELINE_STEPS[activeStep - 1].label}`}
        </span>
      </div>
    </div>
  );
};
