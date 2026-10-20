"use client";

import { useEffect, useState } from "react";
import {
  Scale,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  ShieldCheck,
  ChevronRight,
  Info,
} from "lucide-react";
import { KappaAlignmentMatrix } from "@/components/KappaAlignmentMatrix";
import { api, type AgreementReport } from "@/lib/api";

export default function CalibrationPage() {
  const [report, setReport] = useState<AgreementReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [runningJudge, setRunningJudge] = useState(false);
  const [modelName, setModelName] = useState("heuristic-rubric-judge-v1");
  const [bidirectional, setBidirectional] = useState(true);
  const [temperature, setTemperature] = useState(0.0);
  const [successToast, setSuccessToast] = useState<string | null>(null);

  const fetchReport = async () => {
    setLoading(true);
    try {
      const data = await api.getAgreementReport();
      setReport(data);
    } catch (err) {
      console.error("Failed to fetch agreement report:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, []);

  const handleRunJudge = async () => {
    setRunningJudge(true);
    setSuccessToast(null);
    try {
      const res = await api.runJudgeEvaluation({
        model: modelName,
        temperature: temperature,
        bidirectional_swap: bidirectional,
      });

      setSuccessToast(
        `Dispatched judge evaluation task (${res.total_prompts} prompts evaluated with ${
          bidirectional ? "bidirectional position swap" : "single pass"
        }).`
      );

      // Refresh report after short delay
      setTimeout(() => {
        fetchReport();
      }, 1500);
    } catch (err) {
      console.error("Failed to run judge evaluation:", err);
      alert("Failed to run judge evaluation. Check console.");
    } finally {
      setRunningJudge(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-950/60 border border-violet-800/40 text-violet-400">
            <Scale className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <span>LLM-as-a-Judge Calibration &amp; Agreement Studio</span>
              <span className="rounded-full bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold text-violet-300 border border-violet-500/30">
                IAA Engine
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Measure Cohen&apos;s Kappa, test position-swap consistency, and calibrate automated routing.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={fetchReport}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Report</span>
        </button>
      </div>

      {/* Trigger Judge Evaluation Panel */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div className="space-y-0.5">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <span>Run Automated Judge Assessment</span>
            </h3>
            <p className="text-xs text-slate-400">
              Generate AI judge annotations across the dataset to measure Inter-Annotator Agreement (IAA)
              against humans.
            </p>
          </div>

          <button
            type="button"
            disabled={runningJudge}
            onClick={handleRunJudge}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-5 py-2 text-xs font-semibold text-white shadow-glow hover:brightness-110 disabled:opacity-50 transition-all"
          >
            <Play className={`h-3.5 w-3.5 fill-current ${runningJudge ? "animate-spin" : ""}`} />
            <span>{runningJudge ? "Evaluating Prompts..." : "Trigger Judge Run"}</span>
          </button>
        </div>

        {/* Configuration Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="space-y-1">
            <label className="text-slate-300 font-medium">Judge Model Engine</label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-slate-200 focus:border-violet-500 focus:outline-none"
            >
              <option value="heuristic-rubric-judge-v1">Heuristic Rubric Judge (Offline &amp; Deterministic)</option>
              <option value="gpt-4o-mini">OpenAI gpt-4o-mini (LiteLLM)</option>
              <option value="claude-3-5-haiku-20241022">Anthropic Claude 3.5 Haiku (LiteLLM)</option>
              <option value="ollama/llama3.2:latest">Local Ollama / LLaMA 3.2</option>
            </select>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-slate-300 font-medium">
              <span>Sampling Temperature</span>
              <span className="font-mono text-cyan-400">{temperature}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />
            <span className="text-[10px] text-slate-500">0.0 = Deterministic (recommended for judges)</span>
          </div>

          <div className="space-y-1">
            <label className="text-slate-300 font-medium">Bias Cancellation Protocol</label>
            <label className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 cursor-pointer hover:border-slate-700">
              <input
                type="checkbox"
                checked={bidirectional}
                onChange={(e) => setBidirectional(e.target.checked)}
                className="rounded accent-violet-500"
              />
              <span className="text-slate-200">
                Bidirectional Position-Swap (A/B + B/A)
              </span>
            </label>
            <span className="text-[10px] text-slate-500">
              Tests if flipping candidate order flips the judge&apos;s decision.
            </span>
          </div>
        </div>

        {/* Success Alert */}
        {successToast && (
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-3 text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <span>{successToast}</span>
          </div>
        )}
      </div>

      {/* Main Alignment Matrix / Heatmap */}
      {loading ? (
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-12 text-center space-y-3">
          <RefreshCw className="h-8 w-8 text-violet-400 animate-spin mx-auto" />
          <h3 className="text-base font-semibold text-white">Computing Inter-Annotator Agreement...</h3>
          <p className="text-xs text-slate-400">
            Calculating Cohen&apos;s Kappa, standard error, and 95% confidence intervals.
          </p>
        </div>
      ) : report ? (
        <KappaAlignmentMatrix report={report} onRefresh={fetchReport} isLoading={loading} />
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-800 p-12 text-center text-xs text-slate-500">
          No agreement report available yet. Run the judge evaluation above to generate paired annotations.
        </div>
      )}
    </div>
  );
}
