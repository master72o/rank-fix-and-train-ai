"use client";

import { useMemo } from "react";
import { Scale, CheckCircle2, AlertTriangle, ShieldCheck, HelpCircle } from "lucide-react";
import type { AgreementReport } from "@/lib/api";

interface KappaAlignmentMatrixProps {
  report: AgreementReport;
  onRefresh?: () => void;
  isLoading?: boolean;
}

export function KappaAlignmentMatrix({ report, onRefresh, isLoading = false }: KappaAlignmentMatrixProps) {
  const { labels, matrix } = report.confusion_matrix;

  // Maximum value in matrix for heatmap scaling
  const maxCellVal = useMemo(() => {
    let max = 1;
    for (const row of matrix) {
      for (const val of row) {
        if (val > max) max = val;
      }
    }
    return max;
  }, [matrix]);

  // Color & Badge based on Landis & Koch interpretation
  const getKappaBadge = (kappa: number) => {
    if (kappa >= 0.8) {
      return {
        bg: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
        label: "Almost Perfect",
      };
    }
    if (kappa >= 0.6) {
      return {
        bg: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
        label: "Substantial",
      };
    }
    if (kappa >= 0.4) {
      return {
        bg: "bg-blue-500/10 text-blue-300 border-blue-500/30",
        label: "Moderate",
      };
    }
    if (kappa >= 0.2) {
      return {
        bg: "bg-amber-500/10 text-amber-300 border-amber-500/30",
        label: "Fair",
      };
    }
    return {
      bg: "bg-rose-500/10 text-rose-300 border-rose-500/30",
      label: "Slight / Poor",
    };
  };

  const badge = getKappaBadge(report.cohens_kappa);

  const formatLabel = (l: string) => {
    switch (l) {
      case "candidate_a":
        return "Cand A";
      case "candidate_b":
        return "Cand B";
      case "tie":
        return "Tie";
      case "both_bad":
        return "Both Bad";
      default:
        return l;
    }
  };

  return (
    <div className="space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Cohen's Kappa */}
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-medium">Cohen's Kappa (κ)</span>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${badge.bg}`}>
              {badge.label}
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-white tracking-tight">
              {report.cohens_kappa.toFixed(3)}
            </span>
            <span className="text-xs text-slate-500">
              [CI: {report.ci_lower.toFixed(2)} - {report.ci_upper.toFixed(2)}]
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            Corrected for chance agreement. Benchmark: &gt;0.60 indicates high-fidelity alignment.
          </p>
        </div>

        {/* Observed vs Expected */}
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-medium">Observed Agreement</span>
            <span className="text-[10px] text-slate-500 font-mono">P_o vs P_e</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-cyan-400 tracking-tight">
              {(report.observed_agreement * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500">
              (Exp: {(report.expected_agreement * 100).toFixed(1)}%)
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            Raw agreement percentage over {report.paired_count} double-annotated samples.
          </p>
        </div>

        {/* Position Consistency (A/B vs B/A) */}
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-medium">Position Consistency</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-emerald-400 tracking-tight">
              {(report.position_consistency_rate * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500">Pass Rate</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            Measures resilience against first-position bias under bidirectional swap evaluation.
          </p>
        </div>

        {/* Paired Samples Count */}
        <div className="rounded-2xl border border-slate-800 bg-surface-100 p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-medium">Paired Samples</span>
            <Scale className="h-4 w-4 text-violet-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-violet-400 tracking-tight">
              {report.paired_count}
            </span>
            <span className="text-xs text-slate-500">Prompts</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            Samples co-annotated by both human reviewers and LLM-as-a-judge.
          </p>
        </div>
      </div>

      {/* Confusion Matrix Heatmap Panel */}
      <div className="rounded-2xl border border-slate-800 bg-surface-100 p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <span>Confusion Matrix: Human Reviewers vs. LLM-as-a-Judge</span>
            </h3>
            <p className="text-xs text-slate-400">
              Rows represent Ground Truth (Human Choice), Columns represent AI Judge predictions. Diagonal cells
              indicate perfect alignment.
            </p>
          </div>

          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              disabled={isLoading}
              className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
            >
              {isLoading ? "Recalculating..." : "Refresh IAA"}
            </button>
          )}
        </div>

        {/* Heatmap Grid */}
        <div className="overflow-x-auto py-2">
          <div className="inline-block min-w-full">
            <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-slate-400">
              <span className="w-28 text-right pr-2">Human \ Judge</span>
              {labels.map((colLabel) => (
                <div key={colLabel} className="w-24 text-center font-mono text-cyan-400">
                  {formatLabel(colLabel)}
                </div>
              ))}
            </div>

            {labels.map((rowLabel, rIdx) => {
              const rowValues = matrix[rIdx] || [];
              const rowTotal = rowValues.reduce((a, b) => a + b, 0);

              return (
                <div key={rowLabel} className="flex items-center gap-2 mb-2">
                  <div className="w-28 text-right pr-2 font-mono text-xs font-semibold text-slate-300">
                    {formatLabel(rowLabel)}
                  </div>
                  {rowValues.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const intensity = maxCellVal > 0 ? val / maxCellVal : 0;

                    let bgClass = "bg-slate-800/40 text-slate-400";
                    if (val > 0) {
                      if (isDiagonal) {
                        bgClass =
                          intensity > 0.5
                            ? "bg-emerald-500/30 text-emerald-200 border-emerald-500/50"
                            : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
                      } else {
                        bgClass =
                          intensity > 0.5
                            ? "bg-rose-500/30 text-rose-200 border-rose-500/50"
                            : "bg-rose-500/20 text-rose-300 border-rose-500/30";
                      }
                    }

                    return (
                      <div
                        key={cIdx}
                        className={`w-24 h-14 rounded-xl border flex flex-col items-center justify-center transition-all ${bgClass}`}
                      >
                        <span className="font-mono text-base font-bold">{val}</span>
                        <span className="text-[10px] text-slate-400">
                          {rowTotal > 0 ? `${((val / rowTotal) * 100).toFixed(0)}%` : "0%"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-6 pt-3 border-t border-slate-800/80 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded bg-emerald-500/40 border border-emerald-500/60 inline-block"></span>
            <span>Agreement (Diagonal)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded bg-rose-500/40 border border-rose-500/60 inline-block"></span>
            <span>Disagreement / Position Bias (Off-diagonal)</span>
          </div>
          <div className="flex items-center gap-2">
            <HelpCircle className="h-3.5 w-3.5 text-slate-500" />
            <span>Landis & Koch standard: κ &gt; 0.60 qualifies automated judge auto-acceptance</span>
          </div>
        </div>
      </div>
    </div>
  );
}
