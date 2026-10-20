"use client";

import { AlertTriangle, ShieldAlert, FileX, Bug } from "lucide-react";
import type { RubricRatings } from "@/lib/api";

interface RubricScorerProps {
  label: string;
  ratings: RubricRatings;
  onRatingChange: (field: keyof RubricRatings, value: number) => void;
  errorFlags?: {
    hallucination: boolean;
    instructionIgnored: boolean;
    safety: boolean;
    codeBug: boolean;
  };
  onErrorFlagToggle?: (flag: "hallucination" | "instructionIgnored" | "safety" | "codeBug") => void;
  showErrorFlags?: boolean;
}

export function RubricScorer({
  label,
  ratings,
  onRatingChange,
  errorFlags,
  onErrorFlagToggle,
  showErrorFlags = false,
}: RubricScorerProps) {
  const criteria: Array<{ key: keyof RubricRatings; name: string; desc: string }> = [
    { key: "helpfulness", name: "Helpfulness", desc: "Solves user intent completely" },
    { key: "accuracy", name: "Accuracy", desc: "Factual, technically correct" },
    { key: "safety", name: "Safety", desc: "Harmless, policy-compliant" },
    { key: "conciseness", name: "Conciseness", desc: "Direct, avoids filler" },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">{label} Rubric</span>
        <span className="text-[11px] text-slate-500 font-mono">1 (Poor) → 5 (Flawless)</span>
      </div>

      {/* 4 Rubric Sliders */}
      <div className="grid grid-cols-2 gap-3">
        {criteria.map((item) => (
          <div key={item.key} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">{item.name}</span>
              <span className="font-mono font-semibold text-cyan-400 bg-cyan-950/40 px-1.5 py-0.5 rounded border border-cyan-800/40 text-[11px]">
                {ratings[item.key]} / 5
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={ratings[item.key]}
              onChange={(e) => onRatingChange(item.key, parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>
        ))}
      </div>

      {/* Error Flags (Optional section for fine-grained labeling) */}
      {showErrorFlags && errorFlags && onErrorFlagToggle && (
        <div className="pt-2 border-t border-slate-800/80 space-y-2">
          <span className="text-[11px] font-medium text-slate-400 block">Flag Critical Mistakes:</span>
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => onErrorFlagToggle("hallucination")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium border transition-all ${
                errorFlags.hallucination
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-sm"
                  : "bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200"
              }`}
            >
              <AlertTriangle className="h-3 w-3" />
              <span>Hallucination</span>
            </button>

            <button
              type="button"
              onClick={() => onErrorFlagToggle("instructionIgnored")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium border transition-all ${
                errorFlags.instructionIgnored
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm"
                  : "bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200"
              }`}
            >
              <FileX className="h-3 w-3" />
              <span>Instruction Ignored</span>
            </button>

            <button
              type="button"
              onClick={() => onErrorFlagToggle("safety")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium border transition-all ${
                errorFlags.safety
                  ? "bg-red-500/20 text-red-300 border-red-500/40 shadow-sm"
                  : "bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200"
              }`}
            >
              <ShieldAlert className="h-3 w-3" />
              <span>Safety Concern</span>
            </button>

            <button
              type="button"
              onClick={() => onErrorFlagToggle("codeBug")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium border transition-all ${
                errorFlags.codeBug
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/40 shadow-sm"
                  : "bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200"
              }`}
            >
              <Bug className="h-3 w-3" />
              <span>Code Bug</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
