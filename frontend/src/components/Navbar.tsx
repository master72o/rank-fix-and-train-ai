"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Flame, Layers, Scale, Cpu, Sparkles, Activity } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { href: "/", label: "Overview", icon: Layers },
    { href: "/workbench", label: "RLHF Workbench", icon: Flame },
    { href: "/calibration", label: "Judge Calibration", icon: Scale },
    { href: "/training", label: "DPO Training Hub", icon: Cpu },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-surface-200/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 via-indigo-500 to-cyan-400 p-0.5 shadow-glow group-hover:scale-105 transition-transform">
            <div className="flex h-full w-full items-center justify-center rounded-[10px] bg-surface-200">
              <Sparkles className="h-4 w-4 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white group-hover:text-cyan-300 transition-colors">
                AlignForge
              </span>
              <span className="rounded-full bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold text-violet-400 border border-violet-500/20">
                v1.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Rank, Fix, and Train AI</p>
          </div>
        </Link>

        {/* Nav Links */}
        <nav className="flex items-center gap-1 sm:gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  isActive
                    ? "bg-slate-800 text-cyan-300 shadow-sm border border-slate-700"
                    : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                }`}
              >
                <Icon className={`h-3.5 w-3.5 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Live Engine Indicator */}
        <div className="hidden md:flex items-center gap-2.5 rounded-full bg-slate-900/80 px-3 py-1 border border-slate-800 text-xs">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-slate-400">Engine:</span>
          <span className="font-mono text-emerald-400 font-medium text-[11px]">Active Learning Live</span>
        </div>
      </div>
    </header>
  );
}
