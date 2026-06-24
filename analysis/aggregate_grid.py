"""Aggregate grid results into the headline tables (spec step 6).

Reads ``results/grid/grid_results.csv`` (produced by ``run_grid.py``) and
prints, per source, the recovery rate at each injection size -- and the
**smallest size that recovers** (recovery rate above a threshold). Also reports
the no-intervention floor and, if present, the fresh-run injection baseline.

Usage::

    poetry run python analysis/aggregate_grid.py
    poetry run python analysis/aggregate_grid.py --csv results/grid/grid_results.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SIZE_ORDER = ["word", "sentence", "paragraph"]


def _to_bool(v: str) -> Optional[bool]:
    if v in ("True", "true", "1"):
        return True
    if v in ("False", "false", "0"):
        return False
    return None


def load(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rate(flags: List[Optional[bool]]) -> Tuple[int, int]:
    vals = [b for b in flags if b is not None]
    return sum(1 for b in vals if b), len(vals)


def summarize(rows: List[dict], threshold: float = 0.5) -> None:
    by_cond_size_source: Dict[Tuple[str, str, str], List[Optional[bool]]] = (
        defaultdict(list)
    )
    floor: List[Optional[bool]] = []
    for r in rows:
        rec = _to_bool(str(r.get("recovered")))
        if r["condition"] == "no_intervention":
            floor.append(rec)
        else:
            key = (r["condition"], r["size"], r["source"])
            by_cond_size_source[key].append(rec)

    # No-intervention floor
    f_rec, f_n = _rate(floor)
    print("\n=== No-intervention floor (does it recover on its own?) ===")
    print(f"  recovered {f_rec}/{f_n} runs")

    for condition in ("intervention", "fresh"):
        present = any(k[0] == condition for k in by_cond_size_source)
        if not present:
            continue
        label = (
            "Intervention (collapsed + injection)"
            if condition == "intervention"
            else "Fresh-run injection baseline"
        )
        print(f"\n=== {label}: recovery rate by size x source ===")
        header = f"  {'size':<10} | " + " | ".join(
            f"{s:^12}" for s in ("real", "noise")
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for size in SIZE_ORDER:
            cells = []
            for source in ("real", "noise"):
                rec, n = _rate(
                    by_cond_size_source.get((condition, size, source), [])
                )
                cells.append(f"{rec}/{n}" if n else "  -  ")
            print(f"  {size:<10} | " + " | ".join(f"{c:^12}" for c in cells))

        if condition == "intervention":
            print(
                f"\n  Smallest size that recovers "
                f"(rate > {threshold:.0%}), per source:"
            )
            for source in ("real", "noise"):
                smallest = None
                for size in SIZE_ORDER:
                    rec, n = _rate(
                        by_cond_size_source.get((condition, size, source), [])
                    )
                    if n and rec / n > threshold:
                        smallest = size
                        break
                print(f"    {source:<6}: {smallest or 'none recovered'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/grid/grid_results.csv")
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
