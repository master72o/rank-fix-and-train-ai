"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ArrowLeftRight,
  Sparkles,
  CheckCircle2,
  XCircle,
  Equal,
  Edit3,
  Terminal,
  Zap,
  Info,
  ChevronRight,
  ShieldAlert,
} from "lucide-react";
import { RubricScorer } from "./RubricScorer";
import type { PromptSample, RubricRatings, AnnotationCreatePayload } from "@/lib/api";

interface PreferenceArenaProps {
  sample: PromptSample;
  onAnnotate: (payload: AnnotationCreatePayload) => Promise<void>;
  onSkip?: () => void;
  isSubmitting?: boolean;
}

export function PreferenceArena({ sample, onAnnotate, onSkip, isSubmitting = false }: PreferenceArenaProps) {
  // Swapping visual placement to mitigate left/right position bias
  const [isSwapped, setIsSwapped] = useState(false);

  // User Choice
  const [selectedChoice, setSelectedChoice] = useState<"candidate_a" | "candidate_b" | "tie" | "both_bad" | null>(
    null
  );

  // Error flags
  const [errorFlags, setErrorFlags] = useState({
    hallucination: false,
    instructionIgnored: false,
    safety: false,
    codeBug: false,
  });

  // Rubrics
  const [rubricA, setRubricA] = useState<RubricRatings>({
    helpfulness: 4,
    accuracy: 4,
    safety: 5,
    conciseness: 4,
  });
  const [rubricB, setRubricB] = useState<RubricRatings>({
    helpfulness: 3,
    accuracy: 3,
    safety: 5,
    conciseness: 3,
  });

  // Golden Rewrite
  const [isRewriting, setIsRewriting] = useState(false);
  const [goldenRewrite, setGoldenRewrite] = useState("");
  const [rationale, setRationale] = useState("");

  // Start timer for duration tracking
  const [startTime] = useState<number>(Date.now());

  // Left vs Right visual slot assignment
  const leftCandidate = isSwapped
    ? {
        key: "candidate_b" as const,
        title: "Candidate B",
        text: sample.candidate_b,
        model: sample.candidate_b_model || "Candidate B",
        entropy: sample.candidate_b_logprob_entropy,
        rubric: rubricB,
        setRubric: setRubricB,
      }
    : {
        key: "candidate_a" as const,
        title: "Candidate A",
        text: sample.candidate_a,
        model: sample.candidate_a_model || "Candidate A",
        entropy: sample.candidate_a_logprob_entropy,
        rubric: rubricA,
        setRubric: setRubricA,
      };

  const rightCandidate = isSwapped
    ? {
        key: "candidate_a" as const,
        title: "Candidate A",
        text: sample.candidate_a,
        model: sample.candidate_a_model || "Candidate A",
        entropy: sample.candidate_a_logprob_entropy,
        rubric: rubricA,
        setRubric: setRubricA,
      }
    : {
        key: "candidate_b" as const,
        title: "Candidate B",
        text: sample.candidate_b,
        model: sample.candidate_b_model || "Candidate B",
        entropy: sample.candidate_b_logprob_entropy,
        rubric: rubricB,
        setRubric: setRubricB,
      };

  const toggleErrorFlag = (flag: "hallucination" | "instructionIgnored" | "safety" | "codeBug") => {
    setErrorFlags((prev) => ({ ...prev, [flag]: !prev[flag] }));
  };

  const handleSelect = (choice: "candidate_a" | "candidate_b" | "tie" | "both_bad") => {
    setSelectedChoice(choice);
  };

  const submitCurrentAnnotation = useCallback(async () => {
    if (!selectedChoice) return;

    const durationSeconds = Math.max(1, Math.round((Date.now() - startTime) / 1000));

    const payload: AnnotationCreatePayload = {
      prompt_id: sample.id,
      annotator_id: "human_reviewer_1",
      choice: selectedChoice,
      has_error_hallucination: errorFlags.hallucination,
      has_error_instruction_ignored: errorFlags.instructionIgnored,
      has_error_safety: errorFlags.safety,
      has_error_code_bug: errorFlags.codeBug,
      rubric_candidate_a: rubricA,
      rubric_candidate_b: rubricB,
      golden_rewrite: goldenRewrite.trim() ? goldenRewrite.trim() : null,
      rationale: rationale.trim() ? rationale.trim() : null,
      duration_seconds: durationSeconds,
      presented_candidate_a_as: isSwapped ? "B" : "A",
    };

    await onAnnotate(payload);
  }, [
    selectedChoice,
    sample.id,
    startTime,
    errorFlags,
    rubricA,
    rubricB,
    goldenRewrite,
    rationale,
    isSwapped,
    onAnnotate,
  ]);

  // Global Keyboard Navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in a textarea or input
      const target = e.target as HTMLElement;
      if (target.tagName === "TEXTAREA" || target.tagName === "INPUT") {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          submitCurrentAnnotation();
        }
        return;
      }

      if (e.key === "1") {
        e.preventDefault();
        handleSelect(leftCandidate.key);
      } else if (e.key === "2") {
        e.preventDefault();
        handleSelect(rightCandidate.key);
      } else if (e.key.toLowerCase() === "t") {
        e.preventDefault();
        handleSelect("tie");
      } else if (e.key.toLowerCase() === "x") {
        e.preventDefault();
        handleSelect("both_bad");
      } else if (e.key.toLowerCase() === "s") {
        e.preventDefault();
        setIsSwapped((s) => !s);
      } else if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        setIsRewriting((r) => !r);
      } else if (e.key === "Enter" && selectedChoice) {
        e.preventDefault();
        submitCurrentAnnotation();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [leftCandidate.key, rightCandidate.key, selectedChoice, submitCurrentAnnotation]);

  return (
    <div className="space-y-5">
      {/* Prompt Card */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-5 shadow-lg space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-cyan-950/60 px-2.5 py-1 text-xs font-semibold text-cyan-400 border border-cyan-800/40">
              {sample.task_type.toUpperCase()}
            </span>
            <span className="text-xs text-slate-400 font-mono">ID: {sample.id.slice(0, 8)}</span>
          </div>

          <div className="flex items-center gap-3">
            {/* Priority Score badge */}
            <div className="flex items-center gap-1.5 rounded-full bg-slate-900 px-3 py-1 border border-slate-800">
              <Zap className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-xs text-slate-400">Queue Priority:</span>
              <span className="text-xs font-mono font-bold text-amber-300">
                {(sample.priority_score * 100).toFixed(1)}%
              </span>
            </div>

            {/* Semantic Divergence */}
            {sample.semantic_divergence_score !== null && sample.semantic_divergence_score !== undefined && (
              <div className="flex items-center gap-1.5 rounded-full bg-slate-900 px-3 py-1 border border-slate-800">
                <ArrowLeftRight className="h-3.5 w-3.5 text-violet-400" />
                <span className="text-xs text-slate-400">Divergence:</span>
                <span className="text-xs font-mono font-bold text-violet-300">
                  {sample.semantic_divergence_score.toFixed(3)}
                </span>
              </div>
            )}

            {/* Swap visual position button */}
            <button
              type="button"
              onClick={() => setIsSwapped((s) => !s)}
              title="Swap candidate presentation to counter position bias (Shortcut: S)"
              className="flex items-center gap-1.5 rounded-lg bg-slate-800/80 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
            >
              <ArrowLeftRight className="h-3.5 w-3.5 text-slate-400" />
              <span>Swap Positions [S]</span>
            </button>
          </div>
        </div>

        {/* System Prompt (if present) */}
        {sample.system_prompt && (
          <div className="rounded-lg bg-slate-900/90 p-3 border border-slate-800 text-xs font-mono text-slate-300 space-y-1">
            <span className="text-[10px] uppercase font-semibold text-slate-500 tracking-wider block">
              System Instruction
            </span>
            <p className="whitespace-pre-wrap">{sample.system_prompt}</p>
          </div>
        )}

        {/* User Prompt */}
        <div className="space-y-1">
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block">User Prompt</span>
          <p className="text-sm text-slate-100 font-medium whitespace-pre-wrap leading-relaxed bg-slate-900/40 p-3 rounded-lg border border-slate-800/60">
            {sample.user_prompt}
          </p>
        </div>
      </div>

      {/* Side-by-Side Candidates Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Candidate */}
        <div
          onClick={() => handleSelect(leftCandidate.key)}
          className={`cursor-pointer rounded-2xl border transition-all p-5 flex flex-col justify-between ${
            selectedChoice === leftCandidate.key
              ? "border-emerald-500/80 bg-emerald-950/20 shadow-glow-emerald ring-1 ring-emerald-500"
              : "border-slate-800 bg-surface-100 hover:border-slate-700 hover:bg-slate-900/60"
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-white">{leftCandidate.title}</span>
                {leftCandidate.model && (
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400 font-mono">
                    {leftCandidate.model}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <kbd className="rounded bg-slate-800 px-2 py-0.5 text-[11px] font-mono text-slate-300 border border-slate-700">
                  Key [1]
                </kbd>
                {selectedChoice === leftCandidate.key && (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 animate-in fade-in" />
                )}
              </div>
            </div>

            <div className="rounded-lg bg-surface-200/90 p-4 border border-slate-800/80 text-xs font-mono text-slate-200 whitespace-pre-wrap leading-relaxed max-h-[420px] overflow-y-auto">
              {leftCandidate.text}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80">
            <RubricScorer
              label={leftCandidate.title}
              ratings={leftCandidate.rubric}
              onRatingChange={(f, v) => leftCandidate.setRubric((r) => ({ ...r, [f]: v }))}
            />
          </div>
        </div>

        {/* Right Candidate */}
        <div
          onClick={() => handleSelect(rightCandidate.key)}
          className={`cursor-pointer rounded-2xl border transition-all p-5 flex flex-col justify-between ${
            selectedChoice === rightCandidate.key
              ? "border-emerald-500/80 bg-emerald-950/20 shadow-glow-emerald ring-1 ring-emerald-500"
              : "border-slate-800 bg-surface-100 hover:border-slate-700 hover:bg-slate-900/60"
          }`}
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-white">{rightCandidate.title}</span>
                {rightCandidate.model && (
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400 font-mono">
                    {rightCandidate.model}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <kbd className="rounded bg-slate-800 px-2 py-0.5 text-[11px] font-mono text-slate-300 border border-slate-700">
                  Key [2]
                </kbd>
                {selectedChoice === rightCandidate.key && (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 animate-in fade-in" />
                )}
              </div>
            </div>

            <div className="rounded-lg bg-surface-200/90 p-4 border border-slate-800/80 text-xs font-mono text-slate-200 whitespace-pre-wrap leading-relaxed max-h-[420px] overflow-y-auto">
              {rightCandidate.text}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80">
            <RubricScorer
              label={rightCandidate.title}
              ratings={rightCandidate.rubric}
              onRatingChange={(f, v) => rightCandidate.setRubric((r) => ({ ...r, [f]: v }))}
            />
          </div>
        </div>
      </div>

      {/* Alternative Judgments (Tie, Both Bad, Golden Rewrite) & Error Flags */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Quick Choice Buttons */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => handleSelect(leftCandidate.key)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                selectedChoice === leftCandidate.key
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
              }`}
            >
              <span>{leftCandidate.title} Better</span>
              <kbd className="rounded bg-slate-900 px-1 py-0.5 text-[10px] font-mono text-slate-400">[1]</kbd>
            </button>

            <button
              type="button"
              onClick={() => handleSelect(rightCandidate.key)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                selectedChoice === rightCandidate.key
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
              }`}
            >
              <span>{rightCandidate.title} Better</span>
              <kbd className="rounded bg-slate-900 px-1 py-0.5 text-[10px] font-mono text-slate-400">[2]</kbd>
            </button>

            <button
              type="button"
              onClick={() => handleSelect("tie")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                selectedChoice === "tie"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
              }`}
            >
              <Equal className="h-3.5 w-3.5" />
              <span>Equal / Tie</span>
              <kbd className="rounded bg-slate-900 px-1 py-0.5 text-[10px] font-mono text-slate-400">[T]</kbd>
            </button>

            <button
              type="button"
              onClick={() => handleSelect("both_bad")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                selectedChoice === "both_bad"
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/50"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
              }`}
            >
              <XCircle className="h-3.5 w-3.5" />
              <span>Both Bad</span>
              <kbd className="rounded bg-slate-900 px-1 py-0.5 text-[10px] font-mono text-slate-400">[X]</kbd>
            </button>

            <button
              type="button"
              onClick={() => setIsRewriting((r) => !r)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                isRewriting || goldenRewrite.trim()
                  ? "bg-violet-500/20 text-violet-300 border-violet-500/50"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
              }`}
            >
              <Edit3 className="h-3.5 w-3.5 text-violet-400" />
              <span>Golden Rewrite</span>
              <kbd className="rounded bg-slate-900 px-1 py-0.5 text-[10px] font-mono text-slate-400">[R]</kbd>
            </button>
          </div>

          {/* Submission and Skip */}
          <div className="flex items-center gap-2">
            {onSkip && (
              <button
                type="button"
                onClick={onSkip}
                className="rounded-lg bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-200 border border-slate-700"
              >
                Skip
              </button>
            )}

            <button
              type="button"
              disabled={!selectedChoice || isSubmitting}
              onClick={submitCurrentAnnotation}
              className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 px-4 py-1.5 text-xs font-semibold text-slate-950 hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed shadow-md transition-all"
            >
              <span>{isSubmitting ? "Submitting..." : "Submit & Next [Enter]"}</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Global Error Flags */}
        <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-slate-400">
            <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
            <span>Mark errors in this interaction:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => toggleErrorFlag("hallucination")}
              className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium border ${
                errorFlags.hallucination
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/50"
                  : "bg-slate-800/60 text-slate-400 border-slate-700"
              }`}
            >
              Hallucination
            </button>
            <button
              type="button"
              onClick={() => toggleErrorFlag("instructionIgnored")}
              className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium border ${
                errorFlags.instructionIgnored
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                  : "bg-slate-800/60 text-slate-400 border-slate-700"
              }`}
            >
              Instruction Ignored
            </button>
            <button
              type="button"
              onClick={() => toggleErrorFlag("safety")}
              className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium border ${
                errorFlags.safety
                  ? "bg-red-500/20 text-red-300 border-red-500/50"
                  : "bg-slate-800/60 text-slate-400 border-slate-700"
              }`}
            >
              Safety Issue
            </button>
            <button
              type="button"
              onClick={() => toggleErrorFlag("codeBug")}
              className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium border ${
                errorFlags.codeBug
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/50"
                  : "bg-slate-800/60 text-slate-400 border-slate-700"
              }`}
            >
              Code Bug
            </button>
          </div>
        </div>

        {/* Golden Rewrite Drawer */}
        {(isRewriting || goldenRewrite.trim()) && (
          <div className="pt-3 border-t border-slate-800 space-y-3 animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-violet-400" />
                <span className="text-xs font-semibold text-violet-300">
                  Golden Response Rewrite (Highest Priority for DPO)
                </span>
              </div>
              <span className="text-[11px] text-slate-400">
                Rewritten golden responses automatically outrank all model candidates in DPO training pairs.
              </span>
            </div>

            <textarea
              value={goldenRewrite}
              onChange={(e) => setGoldenRewrite(e.target.value)}
              placeholder="Craft the ideal, factual, well-formatted response that the model should have produced..."
              rows={4}
              className="w-full rounded-xl border border-violet-500/30 bg-surface-200/90 p-3 text-xs font-mono text-slate-100 placeholder-slate-500 focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-400"
            />

            <input
              type="text"
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              placeholder="Optional rationale: why is your rewrite or choice better? (e.g., fixed O(N^2) space bug, corrected factual claim)"
              className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs text-slate-300 placeholder-slate-500 focus:border-slate-700 focus:outline-none"
            />
          </div>
        )}
      </div>
    </div>
  );
}
