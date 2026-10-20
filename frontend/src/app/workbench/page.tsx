"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Flame,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Filter,
  Sparkles,
  RefreshCw,
  Cpu,
  Layers,
} from "lucide-react";
import { PreferenceArena } from "@/components/PreferenceArena";
import { api, type PromptSample, type AnnotationCreatePayload } from "@/lib/api";

export default function WorkbenchPage() {
  const [samples, setSamples] = useState<PromptSample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [minPriority, setMinPriority] = useState<number>(0);
  const [completedCount, setCompletedCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchQueue = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const queue = await api.getQueue(30, minPriority, "pending");
      setSamples(queue);
      setCurrentIndex(0);
    } catch (err: unknown) {
      console.error("Failed to fetch queue:", err);
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to load active learning queue from backend."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, [minPriority]);

  const currentSample = samples[currentIndex];

  const handleAnnotate = async (payload: AnnotationCreatePayload) => {
    setSubmitting(true);
    try {
      await api.submitAnnotation(payload);
      setCompletedCount((c) => c + 1);

      // Remove current sample or advance
      if (currentIndex < samples.length - 1) {
        setCurrentIndex((i) => i + 1);
      } else {
        // Refetch queue if at the end
        await fetchQueue();
      }
    } catch (err) {
      console.error("Failed to submit annotation:", err);
      alert("Error submitting annotation. Check console for details.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    if (currentIndex < samples.length - 1) {
      setCurrentIndex((i) => i + 1);
    }
  };

  return (
    <div className="space-y-6">
      {/* Workbench Navigation Header */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-950/60 border border-cyan-800/40 text-cyan-400">
            <Flame className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <span>RLHF &amp; DPO Annotation Workbench</span>
              <span className="rounded-full bg-cyan-500/10 px-2 py-0.5 text-[10px] font-semibold text-cyan-300 border border-cyan-500/30">
                Active Learning Queue
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Keyboard-first side-by-side preference grading, mistake tagging, and golden rewriting.
            </p>
          </div>
        </div>

        {/* Priority Filter and Queue Counter */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 border border-slate-800">
            <Filter className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-400">Priority:</span>
            <select
              value={minPriority}
              onChange={(e) => setMinPriority(parseFloat(e.target.value))}
              className="bg-transparent text-cyan-400 font-semibold focus:outline-none cursor-pointer"
            >
              <option value="0" className="bg-slate-900 text-slate-200">
                All (0.0+)
              </option>
              <option value="0.3" className="bg-slate-900 text-slate-200">
                Medium+ (0.3+)
              </option>
              <option value="0.5" className="bg-slate-900 text-slate-200">
                High Priority (0.5+)
              </option>
              <option value="0.7" className="bg-slate-900 text-slate-200">
                Critical Uncertainty (0.7+)
              </option>
            </select>
          </div>

          <button
            type="button"
            onClick={fetchQueue}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 font-medium text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Keyboard Shortcuts Floating Cheat Bar */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 px-4 py-2 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-2">
        <span className="font-semibold text-slate-300">Keyboard Shortcuts:</span>
        <div className="flex flex-wrap items-center gap-3 font-mono text-[11px]">
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">1</kbd>
            <span>Candidate A</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">2</kbd>
            <span>Candidate B</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">T</kbd>
            <span>Tie</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">X</kbd>
            <span>Both Bad</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">S</kbd>
            <span>Swap View</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">R</kbd>
            <span>Rewrite</span>
          </div>
          <div>
            <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 border border-slate-700 mr-1">Enter</kbd>
            <span>Submit</span>
          </div>
        </div>

        {samples.length > 0 && (
          <div className="flex items-center gap-2 font-mono text-cyan-400 font-semibold">
            <span>
              Sample {currentIndex + 1} / {samples.length}
            </span>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      {loading ? (
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-12 text-center space-y-3">
          <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin mx-auto" />
          <h3 className="text-base font-semibold text-white">Loading Active Learning Queue...</h3>
          <p className="text-xs text-slate-400">
            Calculating logprob entropies and semantic divergence vectors.
          </p>
        </div>
      ) : samples.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-emerald-500/40 bg-emerald-950/10 p-12 text-center space-y-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 mx-auto">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-white">Queue Empty / All Samples Annotated!</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Every candidate pair in this priority tier has been graded and converted into golden preference
              pairs for Direct Preference Optimization.
            </p>
          </div>
          <div className="pt-2 flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => setMinPriority(0)}
              className="rounded-xl bg-slate-800 px-4 py-2 text-xs font-medium text-slate-200 border border-slate-700 hover:bg-slate-700"
            >
              Reset Filters
            </button>
            <Link
              href="/training"
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-5 py-2 text-xs font-semibold text-white shadow-glow hover:brightness-110"
            >
              <Cpu className="h-4 w-4" />
              <span>Launch DPO Training</span>
            </Link>
          </div>
        </div>
      ) : currentSample ? (
        <PreferenceArena
          sample={currentSample}
          onAnnotate={handleAnnotate}
          onSkip={handleSkip}
          isSubmitting={submitting}
        />
      ) : null}
    </div>
  );
}
