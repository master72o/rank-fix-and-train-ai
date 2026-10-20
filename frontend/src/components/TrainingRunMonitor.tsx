"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import {
  Cpu,
  Trophy,
  CheckCircle2,
  Clock,
  AlertCircle,
  Play,
  Layers,
  Sparkles,
} from "lucide-react";
import type { TrainingRun, MLSystemCapabilities } from "@/lib/api";

interface TrainingRunMonitorProps {
  runs: TrainingRun[];
  capabilities?: MLSystemCapabilities | null;
  onLaunchNewRun: (params: {
    name: string;
    base_model?: string;
    num_train_epochs?: number;
    learning_rate?: number;
    beta?: number;
  }) => Promise<void>;
  isLaunching?: boolean;
}

export function TrainingRunMonitor({
  runs,
  capabilities,
  onLaunchNewRun,
  isLaunching = false,
}: TrainingRunMonitorProps) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    runs.length > 0 ? runs[0].id : null
  );

  const [showLaunchModal, setShowLaunchModal] = useState(false);
  const [runName, setRunName] = useState("alignforge-qwen-dpo-v1");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(0.000005);
  const [beta, setBeta] = useState(0.1);

  const selectedRun = runs.find((r) => r.id === selectedRunId) || runs[0];

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    await onLaunchNewRun({
      name: runName,
      base_model: baseModel,
      num_train_epochs: epochs,
      learning_rate: learningRate,
      beta: beta,
    });
    setShowLaunchModal(false);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner / System Hardware Specs */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-600/20 border border-violet-500/30 text-violet-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Direct Preference Optimization (DPO) Hub</h3>
            <p className="text-xs text-slate-400">
              Fine-tune models directly on golden preference pairs without separate reward modeling.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {capabilities && (
            <div className="hidden sm:flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-1.5 border border-slate-800 text-xs">
              <span className="text-slate-400">Hardware:</span>
              <span className="font-mono text-cyan-300">
                {capabilities.gpu_available
                  ? `${capabilities.gpu_name || "CUDA GPU"} (${capabilities.vram_gb?.toFixed(1) || "?"} GB)`
                  : "CPU / Fallback Mode"}
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowLaunchModal(true)}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-glow hover:brightness-110 transition-all"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            <span>Launch DPO Training</span>
          </button>
        </div>
      </div>

      {/* Main Grid: Runs List + Details View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Runs List (1 Col) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
            <span>Training Runs</span>
            <span>{runs.length} Runs</span>
          </div>

          <div className="space-y-2.5 max-h-[640px] overflow-y-auto">
            {runs.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-800 p-6 text-center text-xs text-slate-500">
                No DPO runs launched yet. Click "Launch DPO Training" above to train on golden preference pairs.
              </div>
            ) : (
              runs.map((run) => {
                const isSelected = selectedRun?.id === run.id;
                return (
                  <div
                    key={run.id}
                    onClick={() => setSelectedRunId(run.id)}
                    className={`cursor-pointer rounded-xl border p-4 transition-all ${
                      isSelected
                        ? "border-violet-500/80 bg-violet-950/20 shadow-glow"
                        : "border-slate-800 bg-surface-100 hover:border-slate-700 hover:bg-slate-900/60"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sm text-white truncate max-w-[180px]">{run.name}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold border ${
                          run.status === "completed"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                            : run.status === "running"
                            ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/30 animate-pulse"
                            : run.status === "failed"
                            ? "bg-rose-500/10 text-rose-400 border-rose-500/30"
                            : "bg-slate-800 text-slate-400 border-slate-700"
                        }`}
                      >
                        {run.status.toUpperCase()}
                      </span>
                    </div>

                    <div className="mt-2 text-xs text-slate-400 font-mono truncate">{run.base_model}</div>

                    <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
                      <span>{run.num_preference_pairs} pairs</span>
                      {run.post_train_win_rate !== null && run.post_train_win_rate !== undefined ? (
                        <span className="font-bold text-emerald-400">
                          {(run.post_train_win_rate * 100).toFixed(1)}% Win Rate
                        </span>
                      ) : (
                        <span>Progress: {run.progress}%</span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Selected Run Details (2 Cols) */}
        {selectedRun && (
          <div className="lg:col-span-2 space-y-5">
            {/* Run Header & Progress */}
            <div className="rounded-2xl border border-slate-800 bg-surface-100 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h4 className="text-base font-bold text-white flex items-center gap-2">
                    <span>{selectedRun.name}</span>
                  </h4>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    Base: {selectedRun.base_model} &bull; Output: {selectedRun.output_adapter_path}
                  </p>
                </div>

                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Clock className="h-3.5 w-3.5" />
                  <span>{new Date(selectedRun.created_at).toLocaleString()}</span>
                </div>
              </div>

              {/* Progress Bar & Phase */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">{selectedRun.phase_message}</span>
                  <span className="font-mono text-cyan-400 font-bold">{selectedRun.progress}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-violet-500 via-indigo-500 to-cyan-400 transition-all duration-500"
                    style={{ width: `${selectedRun.progress}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Post-Training Win Rate Evaluation Card */}
            {selectedRun.post_train_win_rate !== null && selectedRun.post_train_win_rate !== undefined && (
              <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/10 p-5 space-y-3 shadow-glow-emerald">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-emerald-400" />
                    <h4 className="text-sm font-bold text-white">Post-Training Win-Rate Evaluation</h4>
                  </div>
                  <span className="rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-xs font-bold text-emerald-300 border border-emerald-500/40">
                    Validated vs Baseline
                  </span>
                </div>

                <div className="flex items-baseline gap-3">
                  <span className="text-4xl font-extrabold font-mono text-emerald-400">
                    {(selectedRun.post_train_win_rate * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-slate-400">
                    Head-to-head win rate against baseline ({selectedRun.base_model})
                  </span>
                </div>

                {selectedRun.eval_ci_lower !== null && selectedRun.eval_ci_upper !== null && (
                  <p className="text-xs text-slate-400">
                    95% Wilson Confidence Interval:{" "}
                    <span className="font-mono text-slate-200">
                      [{(selectedRun.eval_ci_lower! * 100).toFixed(1)}% &mdash;{" "}
                      {(selectedRun.eval_ci_upper! * 100).toFixed(1)}%]
                    </span>
                    . Evaluated with bidirectional position-swap bias cancellation.
                  </p>
                )}
              </div>
            )}

            {/* Loss Curve (Recharts) */}
            <div className="rounded-2xl border border-slate-800 bg-surface-100 p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-semibold text-white">DPO Loss Trajectory</h4>
                  <p className="text-xs text-slate-400">Cross-entropy preference loss per gradient step</p>
                </div>
                <span className="text-xs font-mono text-violet-400">
                  {selectedRun.loss_history.length} Checkpoints
                </span>
              </div>

              <div className="h-64 w-full pt-2">
                {selectedRun.loss_history.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500">
                    Loss data will stream once training begins.
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={selectedRun.loss_history} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} domain={["auto", "auto"]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#334155",
                          borderRadius: "0.5rem",
                          fontSize: "12px",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={{ r: 2, fill: "#8b5cf6" }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Launch New Run Modal */}
      {showLaunchModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl border border-slate-800 bg-surface-100 p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-violet-400" />
                <span>Launch Direct Preference Optimization (DPO)</span>
              </h3>
              <button
                type="button"
                onClick={() => setShowLaunchModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreate} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Run Name</label>
                <input
                  type="text"
                  required
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 focus:border-violet-500 focus:outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Base Model (HuggingFace ID)</label>
                <input
                  type="text"
                  required
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 focus:border-violet-500 focus:outline-none"
                />
                <span className="text-[11px] text-slate-500">
                  Recommended: Qwen/Qwen2.5-0.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct
                </span>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Epochs</label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value, 10))}
                    className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 focus:border-violet-500 focus:outline-none"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Learning Rate</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 focus:border-violet-500 focus:outline-none"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Beta (DPO)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={beta}
                    onChange={(e) => setBeta(parseFloat(e.target.value))}
                    className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 focus:border-violet-500 focus:outline-none"
                  />
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowLaunchModal(false)}
                  className="rounded-lg bg-slate-800 px-3 py-2 text-slate-300 hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLaunching}
                  className="rounded-lg bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 font-semibold text-white hover:brightness-110 disabled:opacity-50"
                >
                  {isLaunching ? "Launching..." : "Start Training"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
