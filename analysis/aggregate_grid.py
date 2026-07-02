"""Aggregate grid results into the headline tables (spec step 6).

Reads a ``run_grid.py`` summary CSV and prints, per temperature:

  * the **no-intervention floor** -- did the collapsed loop drift back on its
    own, and how far (mean peak displacement)?
  * the **fresh-run baseline** -- how far a big real injection moves a loop that
    has NOT collapsed yet (the control for displacement / basin depth);
  * the **intervention** table -- by injection *size* x *timing*: the
    judge-confirmed recovery rate AND the reframed dose-response endpoints
    (mean peak displacement off the attractor, mean rounds displaced), plus the
    smallest size whose confirmed-recovery rate clears a threshold.

Schema note: this matches what ``run_grid.py`` actually writes -- ``condition``
in {floor, inject, fresh}, sizes {paragraph, output_sized, skim_half}, no
``source`` column (real snippets only; noise is Kong et al.'s settled result).

Usage::

    poetry run python analysis/aggregate_grid.py --csv results/study/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Canonical display order; any unrecognised value is appended after these.
SIZE_ORDER = ["paragraph", "output_sized", "skim_half"]
TIMING_ORDER = ["after_collapse", "early_5", "every_5"]


def _to_bool(v: object) -> Optional[bool]:
    s = str(v)
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    return None


def _to_float(v: object) -> Optional[float]:
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def load(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rate(flags: List[Optional[bool]]) -> Tuple[int, int]:
    vals = [b for b in flags if b is not None]
    return sum(1 for b in vals if b), len(vals)


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _ordered(values: set, order: List[str]) -> List[str]:
    known = [v for v in order if v in values]
    extra = sorted(v for v in values if v not in order)
    return known + extra


def _fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "  -  "


def summarize(rows: List[dict], threshold: float = 0.5) -> None:
    floor = [r for r in rows if r.get("condition") == "floor"]
    inject = [r for r in rows if r.get("condition") == "inject"]
    fresh = [r for r in rows if r.get("condition") == "fresh"]

    # ---- No-intervention floor -------------------------------------------
    f_rec, f_n = _rate([_to_bool(r.get("recovered")) for r in floor])
    f_disp = _mean([_to_float(r.get("peak_displacement")) for r in floor])
    coll = [r for r in floor if _to_float(r.get("collapse_onset")) is not None]
    print("\n=== No-intervention floor (does it drift back on its own?) ===")
    print(f"  runs: {len(floor)}   collapsed: {len(coll)}/{len(floor)}")
    print(f"  recovered on its own: {f_rec}/{f_n}")
    print(f"  mean peak displacement from attractor: {_fmt(f_disp)}")

    # ---- Fresh-run baseline (control) ------------------------------------
    if fresh:
        print("\n=== Fresh-run baseline (inject into a NOT-collapsed loop) ===")
        sizes = _ordered({r.get("size", "") for r in fresh}, SIZE_ORDER)
        print(f"  {'size':<13} | {'peak disp':^11} | {'disp rounds':^11}")
        print("  " + "-" * 41)
        for size in sizes:
            cells = [r for r in fresh if r.get("size") == size]
            disp = _mean([_to_float(r.get("peak_displacement")) for r in cells])
            dr = _mean([_to_float(r.get("displaced_rounds")) for r in cells])
            print(f"  {size:<13} | {_fmt(disp):^11} | {_fmt(dr):^11}")

    # ---- Intervention (collapsed + injection) ----------------------------
    if not inject:
        return
    sizes = _ordered({r.get("size", "") for r in inject}, SIZE_ORDER)
    timings = _ordered({r.get("timing", "") for r in inject}, TIMING_ORDER)

    def cell(size: str, timing: str) -> List[dict]:
        return [
            r for r in inject
            if r.get("size") == size and r.get("timing") == timing
        ]

    print("\n=== Intervention: judge-confirmed recovery rate (size x timing) ===")
    _print_table(
        sizes, timings,
        lambda s, t: _rate_str(
            [_to_bool(r.get("recovered_confirmed")) for r in cell(s, t)]
        ),
    )

    print("\n=== Dose-response: mean PEAK displacement off attractor ===")
    _print_table(
        sizes, timings,
        lambda s, t: _fmt(
            _mean([_to_float(r.get("peak_displacement")) for r in cell(s, t)])
        ),
    )

    print("\n=== Dose-response: mean ROUNDS displaced (transient length) ===")
    _print_table(
        sizes, timings,
        lambda s, t: _fmt(
            _mean([_to_float(r.get("displaced_rounds")) for r in cell(s, t)])
        ),
    )

    print(
        f"\n  Smallest size with confirmed-recovery rate > {threshold:.0%}, "
        f"per timing:"
    )
    for timing in timings:
        smallest = None
        for size in sizes:
            rec, n = _rate(
                [_to_bool(r.get("recovered_confirmed")) for r in cell(size, timing)]
            )
            if n and rec / n > threshold:
                smallest = size
                break
        print(f"    {timing:<15}: {smallest or 'none recovered'}")


def _rate_str(flags: List[Optional[bool]]) -> str:
    rec, n = _rate(flags)
    return f"{rec}/{n}" if n else "  -  "


def _print_table(sizes, timings, value_fn) -> None:
    header = f"  {'size':<13} | " + " | ".join(f"{t:^15}" for t in timings)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for size in sizes:
        cells = [value_fn(size, t) for t in timings]
        print(f"  {size:<13} | " + " | ".join(f"{c:^15}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/study/summary.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    rows = load(args.csv)
    temps = sorted({r.get("temp", "") for r in rows if r.get("temp", "")})
    if len(temps) > 1:
        for t in temps:
            print(f"\n############### TEMPERATURE {t} ###############")
            summarize(
                [r for r in rows if r.get("temp", "") == t],
                threshold=args.threshold,
            )
    else:
        summarize(rows, threshold=args.threshold)


if __name__ == "__main__":
    main()
