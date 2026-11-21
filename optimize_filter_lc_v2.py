#!/usr/bin/env python3
"""Constrained, adaptive optimization of the team's symmetric 5-pole LC filter.

Run from Project_Sieu_Cao_Tan--main:
    /usr/local/bin/python3 optimize_filter_lc_v2.py

Dependencies: numpy scipy matplotlib + local components.py/netlist.py/solver.py.
Does NOT modify filter_lc.net; uses the team's own S-matrix solver.

Algorithm:
  1. Differential Evolution on a coarse grid (global feasibility search).
  2. SLSQP with explicit per-frequency constraints (local refinement).
  3. Independently re-evaluate on a dense grid and adaptively add worst
     offending frequencies to the local grid, if necessary.
  4. Export/round-trip check netlist, plots, CSV, convergence and JSON metrics.

Caveat: dense-grid validation is NOT a mathematical guarantee between samples.
The objective specs below are demonstration targets, not professor-issued targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from netlist import read_netlist
from solver import solve_netlist_at_freq
from components import to_si

HERE = Path(__file__).resolve().parent
PARAMS = {"C1": ("C", "p", 1e-12), "L2": ("L", "n", 1e-9),
          "C3": ("C", "p", 1e-12), "L4": ("L", "n", 1e-9),
          "C5": ("C", "p", 1e-12)}
GROUPS = {"C1": 0, "C5": 0, "L2": 1, "L4": 1, "C3": 2}
GROUP_NAMES = ("C1=C5", "L2=L4", "C3")
# Wider L range than v1 (v1's optimum sat at the 1.45 upper bound).
BOUNDS = ((0.60, 1.60), (0.60, 2.00), (0.60, 1.60))
PB_LO, PB_HI = 0.10e9, 0.75e9
SB_LO, SB_HI = 1.20e9, 1.60e9
PB_MAX_DB, SB_MIN_DB = 0.35, 22.00
LOG_FLOOR = 1e-14


def spectrum(net, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return insertion loss and return loss in dB for a 2-port circuit."""
    il, rl = [], []
    for f in freqs:
        s = solve_netlist_at_freq(net, float(f), method="matrix")
        if s.shape != (2, 2) or not np.isfinite(s).all():
            raise ValueError(f"Invalid two-port S matrix at {f / 1e9:g} GHz")
        il.append(-20 * np.log10(max(abs(s[1, 0]), LOG_FLOOR)))
        rl.append(-20 * np.log10(max(abs(s[0, 0]), LOG_FLOOR)))
    return np.array(il), np.array(rl)


def original_values(net) -> dict[str, float]:
    elements = {e["name"]: e for e in net.elements}
    missing = set(PARAMS) - elements.keys()
    if missing:
        raise ValueError(f"Required LC components missing: {sorted(missing)}")
    for name, (key, _, _) in PARAMS.items():
        if elements[name]["type"] != ("CAP" if key == "C" else "IND"):
            raise ValueError(f"{name} is not the expected {key} element")
    result = {name: float(to_si(elements[name]["params"][key]))
              for name, (key, _, _) in PARAMS.items()}
    # np.isclose() default atol=1e-8 would incorrectly accept any pF/nH values!
    for a, b in (("C1", "C5"), ("L2", "L4")):
        if not np.isclose(result[a], result[b], rtol=1e-6, atol=0.0):
            raise ValueError(f"Input filter must be symmetric: {a} != {b}")
    if any(v <= 0 or not np.isfinite(v) for v in result.values()):
        raise ValueError("Component values must be finite and positive")
    return result


def set_values(net, factors: np.ndarray, originals: dict[str, float]) -> None:
    """Update parsed netlist in-place; avoids reparsing on every optimization evaluation."""
    for elem in net.elements:
        name = elem["name"]
        if name in GROUPS:
            key, suffix, scale = PARAMS[name]
            value = originals[name] * float(factors[GROUPS[name]])
            elem["params"][key] = f"{value/scale:.12g}{suffix}"


def soft_score(p: np.ndarray, s: np.ndarray, target_p: float, target_s: float) -> float:
    """Global-search score: minimize spec violations plus mild ripple and drift."""
    vp = np.maximum(p - target_p, 0.0)
    vs = np.maximum(target_s - s, 0.0)
    return float(25 * np.mean(vp ** 2) + 3 * np.mean(vs ** 2)
                 + 0.01 * np.mean(np.maximum(p, 0) ** 2))


class Evaluator:
    def __init__(self, net, originals, pass_f, stop_f):
        self.net = net
        self.originals = originals
        self.pass_f = np.asarray(pass_f, dtype=float)
        self.stop_f = np.asarray(stop_f, dtype=float)
        self.calls = 0
        self.solve_attempts = 0
        self._last_x = None
        self._last_values = None

    def frequencies(self, pb, sb):
        self.pass_f = np.asarray(sorted(set(np.r_[self.pass_f, pb])), dtype=float)
        self.stop_f = np.asarray(sorted(set(np.r_[self.stop_f, sb])), dtype=float)
        self._last_x = None

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return self._last_values
        self.calls += 1
        self.solve_attempts += len(self.pass_f) + len(self.stop_f)
        set_values(self.net, x, self.originals)
        p, _ = spectrum(self.net, self.pass_f)
        s, _ = spectrum(self.net, self.stop_f)
        self._last_x = x.copy()
        self._last_values = (p, s)
        return p, s


def export_netlist(source: Path, target: Path, params: dict[str, str]) -> None:
    """Only replace named C/L tokens, preserving all original wiring/comments/newlines."""
    lines = []
    replaced = set()
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        match = re.match(r"^\s*(CAP|IND)\s+(\S+)\s+", line, flags=re.IGNORECASE)
        if match:
            name = match.group(2)
            if name in params:
                key = PARAMS[name][0]
                pat = r"\b" + key + r"=[^\s;#!]+"
                line, n = re.subn(pat, f"{key}={params[name]}", line, count=1,
                                  flags=re.IGNORECASE)
                if n != 1:
                    raise ValueError(f"Could not update {name} on its netlist line")
                replaced.add(name)
        lines.append(line)
    if replaced != set(PARAMS):
        raise ValueError(f"Output incomplete; updated {replaced}")
    target.write_text("".join(lines), encoding="utf-8")


def assess(p, s, pass_limit=PB_MAX_DB, stop_limit=SB_MIN_DB):
    return {"peak_pass_loss_db": float(np.max(p)),
            "min_stop_attenuation_db": float(np.min(s)),
            "pass_violation_db": float(max(0, np.max(p) - pass_limit)),
            "stop_violation_db": float(max(0, stop_limit - np.min(s))),
            "feasible": bool(np.max(p) <= pass_limit + 1e-9
                             and np.min(s) >= stop_limit - 1e-9)}


def add_critical_points(freqs, loss, mode, n=3):
    """Include the true observed extremum and its close neighbors."""
    ordering = np.argsort(-loss if mode == "pass" else loss)
    chosen = []
    for i in ordering:
        if all(abs(int(i) - j) > 3 for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n:
            break
    candidates = set()
    for i in chosen:
        candidates.update(float(freqs[j]) for j in (max(0, i-1), i, min(len(freqs)-1, i+1)))
    return np.array(sorted(candidates))


def run_optimizer(args):
    t0 = time.perf_counter()
    source = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find input netlist: {source}")
    net = read_netlist(str(source))
    originals = original_values(net)
    p_f = np.linspace(PB_LO, PB_HI, args.coarse_points)
    s_f = np.linspace(SB_LO, SB_HI, args.coarse_points)
    dense_p = np.linspace(PB_LO, PB_HI, args.verify_points)
    dense_s = np.linspace(SB_LO, SB_HI, args.verify_points)
    # Safety guards: actual design targets are PB_MAX_DB and SB_MIN_DB.
    guard_pb, guard_sb = PB_MAX_DB - args.pass_guard, SB_MIN_DB + args.stop_guard
    if guard_pb <= 0 or args.stop_guard < 0:
        raise ValueError("Unreasonable guard settings")
    ev = Evaluator(net, originals, p_f, s_f)
    history = []
    de_iter = [0]

    def evaluate_safe(x):
        try:
            return ev.evaluate(x)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return (np.full(len(ev.pass_f), 1e4),
                    np.full(len(ev.stop_f), -1e4))

    def global_objective(x):
        p, s = evaluate_safe(x)
        loss = soft_score(p, s, guard_pb, guard_sb)
        return loss + 0.002 * float(np.dot(np.log(x), np.log(x)))

    def de_callback(xk, convergence):
        de_iter[0] += 1
        history.append(("DE", de_iter[0], float(global_objective(xk))))
        if not args.quiet and (de_iter[0] == 1 or de_iter[0] % 5 == 0):
            print(f"  DE iteration {de_iter[0]:>2}: score={history[-1][2]:.6g}", flush=True)
        return False

    print("=== LC FILTER OPTIMIZER v2 | TEAM S-MATRIX SOLVER ===", flush=True)
    print(f"Netlist: {source.name} | DE + constrained SLSQP | seed={args.seed}", flush=True)
    print(f"Required: PB <= {PB_MAX_DB:.3f} dB; SB >= {SB_MIN_DB:.3f} dB", flush=True)
    print(f"Internal guard: PB <= {guard_pb:.3f} dB; SB >= {guard_sb:.3f} dB", flush=True)
    print(f"Bounds (scaled): {dict(zip(GROUP_NAMES, BOUNDS))}", flush=True)

    result = differential_evolution(global_objective, BOUNDS, seed=args.seed,
                                    maxiter=args.maxiter, popsize=args.popsize,
                                    tol=1e-6, polish=False, workers=1,
                                    callback=de_callback)
    seeds = [np.asarray(result.x), np.ones(3)]
    best_x = seeds[0].copy()
    local_messages = []

    def local_objective(x):
        p, _ = evaluate_safe(x)
        # Once feasible, reduce ripple and minimize departures from initial parts.
        return float(0.04 * np.mean(np.maximum(p, 0) ** 2)
                     + 0.01 * np.sum(np.log(x) ** 2))

    def constraints(x):
        p, s = evaluate_safe(x)
        return np.r_[guard_pb - p, s - guard_sb]

    def local_search(x0, label):
        r = minimize(local_objective, x0=np.asarray(x0), method="SLSQP",
                     bounds=BOUNDS, constraints={"type": "ineq", "fun": constraints},
                     options={"maxiter": args.local_maxiter, "ftol": 1e-9,
                              "disp": False})
        p, s = evaluate_safe(r.x)
        feasible = bool(np.min(constraints(r.x)) >= -2e-5)
        local_messages.append({"stage": label, "success": bool(r.success),
                               "message": str(r.message), "feasible_guard": feasible,
                               "iterations": int(r.nit)})
        return r.x, feasible

    print("Global search complete; starting constrained local refinement...", flush=True)
    feasible_candidates = []
    for i, x in enumerate(seeds):
        candidate, ok = local_search(x, f"initial-{i+1}")
        if ok:
            feasible_candidates.append(candidate)
    if feasible_candidates:
        best_x = min(feasible_candidates, key=local_objective)
    else:
        # No feasible design found yet; we will report this explicitly.
        print("WARNING: initial SLSQP did not reach all guarded limits; trying dense refinement.", flush=True)

    best_metrics = None
    pass_dense = stop_dense = None
    for round_idx in range(args.refine_rounds + 1):
        set_values(net, best_x, originals)
        pass_dense, _ = spectrum(net, dense_p)
        stop_dense, _ = spectrum(net, dense_s)
        best_metrics = assess(pass_dense, stop_dense)
        guard_ok = (np.max(pass_dense) <= guard_pb + 1e-6
                    and np.min(stop_dense) >= guard_sb - 1e-6)
        print(f"  Dense check {round_idx}: PB max={best_metrics['peak_pass_loss_db']:.6f} dB, "
              f"SB min={best_metrics['min_stop_attenuation_db']:.6f} dB, "
              f"nominal={'PASS' if best_metrics['feasible'] else 'FAIL'}, "
              f"guard={'PASS' if guard_ok else 'FAIL'}", flush=True)
        if guard_ok or round_idx == args.refine_rounds:
            break
        # Refine the coarse grid where dense verification finds the worst points.
        ev.frequencies(add_critical_points(dense_p, pass_dense, "pass"),
                       add_critical_points(dense_s, stop_dense, "stop"))
        x_refined, ok = local_search(best_x, f"adaptive-{round_idx+1}")
        if ok:
            best_x = x_refined
        else:
            # Try penalty re-search from best available point, then constrained local.
            relaxed = minimize(global_objective, best_x, method="L-BFGS-B",
                               bounds=BOUNDS, options={"maxiter": 100})
            x_refined, ok = local_search(relaxed.x, f"adaptive-retry-{round_idx+1}")
            if ok or global_objective(x_refined) < global_objective(best_x):
                best_x = x_refined
        history.append(("adaptive", round_idx+1, float(global_objective(best_x))))

    set_values(net, best_x, originals)
    # Re-evaluate dense metrics for final candidate, independently of cached values.
    pass_dense, _ = spectrum(net, dense_p)
    stop_dense, _ = spectrum(net, dense_s)
    final_metrics = assess(pass_dense, stop_dense)
    final_guard_ok = (max(pass_dense) <= guard_pb + 1e-6
                      and min(stop_dense) >= guard_sb - 1e-6)
    final_source = read_netlist(str(source))
    baseline_p, _ = spectrum(final_source, dense_p)
    baseline_s, _ = spectrum(final_source, dense_s)
    base_metrics = assess(baseline_p, baseline_s)

    value_map = {e["name"]: e["params"][PARAMS[e["name"]][0]]
                 for e in net.elements if e["name"] in PARAMS}
    target = out / "filter_lc_optimized_v2.net"
    export_netlist(source, target, value_map)
    roundtrip = read_netlist(str(target))
    # Use DIFFERENT netlist object and dense grid to verify the export.
    check_p, _ = spectrum(roundtrip, dense_p)
    check_s, _ = spectrum(roundtrip, dense_s)
    if (np.max(np.abs(pass_dense - check_p)) > 1e-7
            or np.max(np.abs(stop_dense - check_s)) > 1e-7):
        raise AssertionError("Exported netlist does NOT reproduce optimized response")

    chart_f = np.linspace(0.05e9, 2.0e9, args.plot_points)
    base_il, base_rl = spectrum(final_source, chart_f)
    opt_il, opt_rl = spectrum(roundtrip, chart_f)
    with (out / "filter_lc_before_after_v2.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_GHz", "insertion_loss_before_dB", "insertion_loss_after_dB",
                    "return_loss_before_dB", "return_loss_after_dB"])
        for fr, a, b, c, d in zip(chart_f, base_il, opt_il, base_rl, opt_rl):
            w.writerow([f"{fr/1e9:.9g}", f"{a:.9g}", f"{b:.9g}",
                        f"{c:.9g}", f"{d:.9g}"])

    # Dense validation per specification band, including all grid points.
    with (out / "dense_verification_v2.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "frequency_GHz", "loss_before_dB", "loss_after_dB"])
        for band, fr, a, b in (("pass", dense_p, baseline_p, pass_dense),
                              ("stop", dense_s, baseline_s, stop_dense)):
            for freq, orig, new in zip(fr, a, b):
                w.writerow([band, f"{freq/1e9:.12g}", f"{orig:.10g}", f"{new:.10g}"])
    with (out / "convergence_v2.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "step", "score"])
        w.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axs[0].plot(chart_f/1e9, base_il, label="Before", linewidth=1.6)
    axs[0].plot(chart_f/1e9, opt_il, label="Optimized v2", linewidth=1.8)
    axs[0].axhline(PB_MAX_DB, ls="--", color="tab:green", lw=1, label="PB limit")
    axs[0].axhline(SB_MIN_DB, ls="--", color="tab:red", lw=1, label="SB limit")
    axs[0].set(ylabel="Insertion loss (dB)", title="LC filter: constrained optimization | team S-matrix solver")
    axs[0].set_ylim(-1, 55)
    axs[0].legend(loc="upper left")
    axs[1].plot(chart_f/1e9, base_rl, label="Before", linewidth=1.6)
    axs[1].plot(chart_f/1e9, opt_rl, label="Optimized v2", linewidth=1.8)
    axs[1].set(xlabel="Frequency (GHz)", ylabel="Input return loss (dB)")
    axs[1].set_ylim(-2, 50)
    axs[1].legend(loc="upper right")
    for ax in axs:
        ax.axvspan(PB_LO/1e9, PB_HI/1e9, alpha=.07, color="tab:green")
        ax.axvspan(SB_LO/1e9, SB_HI/1e9, alpha=.06, color="tab:red")
        ax.grid(alpha=.25)
    fig.savefig(out / "filter_lc_before_after_v2.png", dpi=170)
    plt.close(fig)

    runtime = time.perf_counter() - t0
    summary = {"input": str(source), "model": "ideal 5th-order symmetric LC ladder",
               "method": "DE global + SLSQP explicit constraints + adaptive dense-grid validation",
               "specification_source": "demonstration values, not professor's direct requirement",
               "targets": {"passband_GHz": [PB_LO/1e9, PB_HI/1e9],
                           "stopband_GHz": [SB_LO/1e9, SB_HI/1e9],
                           "passband_limit_dB": PB_MAX_DB, "stopband_attenuation_dB": SB_MIN_DB,
                           "pass_guard_dB": args.pass_guard, "stop_guard_dB": args.stop_guard},
               "bounds": dict(zip(GROUP_NAMES, BOUNDS)),
               "scale_factors": dict(zip(GROUP_NAMES, map(float, best_x))),
               "original_SI": originals,
               "optimized_SI": {k: float(to_si(value_map[k])) for k in PARAMS},
               "baseline_dense": base_metrics, "optimized_dense": final_metrics,
               "guard_feasible_on_dense_grid": bool(final_guard_ok),
               "verification_points_per_band": int(args.verify_points),
               "evaluations": int(ev.calls), "approx_frequency_solves": int(ev.solve_attempts),
               "DE_iterations": int(de_iter[0]), "local_stages": local_messages,
               "runtime_seconds": runtime,
               "export_round_trip": "PASS", "continuous_band_guarantee": False}
    (out / "optimization_metrics_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== FINAL RESULT (DENSE VALIDATION) ===", flush=True)
    print(f"PB peak: {base_metrics['peak_pass_loss_db']:.6f} -> {final_metrics['peak_pass_loss_db']:.6f} dB", flush=True)
    print(f"SB min : {base_metrics['min_stop_attenuation_db']:.6f} -> {final_metrics['min_stop_attenuation_db']:.6f} dB", flush=True)
    print(f"Design specs: {'PASS' if final_metrics['feasible'] else 'FAIL'} | "
          f"Guard: {'PASS' if final_guard_ok else 'FAIL'}", flush=True)
    print("Group scale factors:", dict(zip(GROUP_NAMES, np.round(best_x, 8))), flush=True)
    for name in PARAMS:
        print(f"  {name}: {originals[name]:.7g} -> {float(to_si(value_map[name])):.7g} SI", flush=True)
    print(f"Evaluations: {ev.calls}; approximately {ev.solve_attempts} frequency solves; "
          f"elapsed {runtime:.1f} s", flush=True)
    print("PASS: saved optimized netlist reproduces the dense response", flush=True)
    print("Output:", out, flush=True)
    if not final_metrics["feasible"]:
        print("WARNING: Spec not met; output is best effort and must NOT be labeled feasible.", flush=True)
    print(f"NOTE: {args.verify_points}-point-per-band validation does not prove constraints between samples.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(HERE/"filter_lc.net"))
    parser.add_argument("--output", default=str(HERE/"results_optimization_v2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=24, help="Differential evolution max iterations")
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument("--local-maxiter", type=int, default=150)
    parser.add_argument("--coarse-points", type=int, default=11,
                        help="Number of initial samples PER band")
    parser.add_argument("--verify-points", type=int, default=1001,
                        help="Dense validation samples PER band")
    parser.add_argument("--plot-points", type=int, default=501)
    parser.add_argument("--refine-rounds", type=int, default=3)
    parser.add_argument("--pass-guard", type=float, default=0.02)
    parser.add_argument("--stop-guard", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if min(args.coarse_points, args.verify_points, args.plot_points) < 3:
        parser.error("coarse, verify and plot points must be >= 3")
    if args.maxiter < 1 or args.popsize < 2 or args.refine_rounds < 0:
        parser.error("invalid optimization iteration/population settings")
    run_optimizer(args)


if __name__ == "__main__":
    main()
