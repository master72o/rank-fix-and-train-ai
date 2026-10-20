"use client";

import { useEffect, useState } from "react";
import {
  Cpu,
  Layers,
  Sparkles,
  RefreshCw,
  Eye,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Trophy,
} from "lucide-react";
import { TrainingRunMonitor } from "@/components/TrainingRunMonitor";
import {
  api,
  type TrainingRun,
  type MLSystemCapabilities,
  type PreferencePair,
} from "@/lib/api";

export default function TrainingPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [capabilities, setCapabilities] = useState<MLSystemCapabilities | null>(null);
  const [pairs, setPairs] = useState<PreferencePair[]>([]);
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [showPairsDrawer, setShowPairsDrawer] = useState(false);

  const fetchTrainingData = async () => {
    setLoading(true);
    try {
      const [runsData, capData, pairsData] = await Promise.allSettled([
        api.getTrainingRuns(),
        api.getCapabilities(),
        api.getPreferencePairs(),
      ]);

      if (runsData.status === "fulfilled") {
        setRuns(runsData.value);
      }
      if (capData.status === "fulfilled") {
        setCapabilities(capData.value);
      }
      if (pairsData.status === "fulfilled") {
        setPairs(pairsData.value);
      }
    } catch (err) {
      console.error("Failed to load training data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrainingData();
  }, []);

  const handleLaunchRun = async (params: {
    name: string;
    base_model?: string;
    num_train_epochs?: number;
    learning_rate?: number;
    beta?: number;
  }) => {
    setLaunching(true);
    try {
      const newRun = await api.launchTraining(params);
      setRuns((prev) => [newRun, ...prev]);

      // Poll periodically while running
      const interval = setInterval(async () => {
        try {
          const updated = await api.getTrainingRun(newRun.id);
          setRuns((prev) => prev.map((r) => (r.id === updated.id ? updated : r)));
          if (updated.status === "completed" || updated.status === "failed") {
            clearInterval(interval);
          }
        } catch {
          clearInterval(interval);
        }
      }, 2500);
    } catch (err) {
      console.error("Failed to launch training run:", err);
      alert("Failed to launch DPO training run. Check console.");
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-950/60 border border-violet-800/40 text-violet-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <span>Direct Preference Optimization (DPO) Hub</span>
              <span className="rounded-full bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold text-violet-300 border border-violet-500/30">
                QLoRA Engine
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Train smarter models on curated golden preference pairs and evaluate head-to-head win rate.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowPairsDrawer((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-cyan-300 hover:bg-slate-700 transition-colors border border-slate-700"
          >
            <Eye className="h-3.5 w-3.5" />
            <span>
              {showPairsDrawer ? "Hide Preference Pairs" : `Preview Preference Pairs (${pairs.length})`}
            </span>
          </button>

          <button
            type="button"
            onClick={fetchTrainingData}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Preference Pairs Preview Drawer */}
      {showPairsDrawer && (
        <div className="rounded-2xl border border-cyan-500/30 bg-surface-100 p-6 space-y-4 animate-in fade-in">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-cyan-400" />
                <span>Extracted Golden Preference Pairs ({pairs.length} Total)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Generated from human reviews and rewrites. Ties and disputes are dropped to preserve gradient
                signal purity.
              </p>
            </div>
            <span className="text-xs text-cyan-400 font-mono">DPO Dataset Ready</span>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {pairs.length === 0 ? (
              <div className="text-xs text-slate-500 text-center py-6">
                No preference pairs available yet. Grade prompts in the RLHF Workbench first!
              </div>
            ) : (
              pairs.map((pair, idx) => (
                <div key={idx} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-300">Prompt: {pair.user_prompt}</span>
                    <span className="rounded bg-violet-500/20 px-2 py-0.5 text-[10px] text-violet-300 font-mono border border-violet-500/30">
                      {pair.source}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                    <div className="rounded-lg bg-emerald-950/20 border border-emerald-500/30 p-3 space-y-1">
                      <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
                        &check; Chosen Response
                      </span>
                      <p className="text-slate-200 whitespace-pre-wrap max-h-32 overflow-y-auto">
                        {pair.chosen}
                      </p>
                    </div>

                    <div className="rounded-lg bg-rose-950/20 border border-rose-500/30 p-3 space-y-1">
                      <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider block">
                        &cross; Rejected Response
                      </span>
                      <p className="text-slate-400 whitespace-pre-wrap max-h-32 overflow-y-auto">
                        {pair.rejected}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Training Monitor Component */}
      <TrainingRunMonitor
        runs={runs}
        capabilities={capabilities}
        onLaunchNewRun={handleLaunchRun}
        isLaunching={launching}
      />
    </div>
  );
}
