#!/usr/bin/env python3
"""
Extract training times from experiment logs.

Reads two types of logs for each case:
  1. experiments/<case>/inv_phy_log.log  — first-order optimization (inverse physics)
  2. experiments_optimization/<case>/optimize_cma_log.log — CMA optimization

For logs with multiple runs, only the LAST run is used.
Outputs a CSV with times in minutes.

Usage:
    python3 extract_training_times.py [--output OUTPUT_CSV]
"""

import os
import re
import csv
import argparse
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
EXPERIMENTS_OPT_DIR = os.path.join(BASE_DIR, "experiments_optimization")


def parse_timestamp(line):
    """Extract datetime from a log line like '2025-02-22 00:35:10,106 [...]'."""
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}", line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return None


def parse_inv_phy_log(filepath):
    """
    Parse inv_phy_log.log for the LAST training run.
    Returns (duration_min, first_iter, last_iter, start_dt, end_dt) or None.
    """
    if not os.path.isfile(filepath):
        return None

    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Find the last training run ---
    # The last occurrence of "Iteration: 1," marks the start of the last run.
    # (Iteration 0 is skipped because it includes one-time visualization overhead.)
    last_run_start_idx = None
    for i, line in enumerate(lines):
        if "[Train]: Iteration: 1," in line:
            last_run_start_idx = i

    if last_run_start_idx is None:
        # Fallback: use "Iteration: 0" if no Iteration 1 exists
        for i, line in enumerate(lines):
            if "[Train]: Iteration: 0," in line:
                last_run_start_idx = i

    if last_run_start_idx is None:
        return None

    # --- Find the last training iteration in the entire file ---
    last_train_idx = None
    for i, line in enumerate(lines):
        if "[Train]: Iteration:" in line:
            last_train_idx = i

    if last_train_idx is None or last_train_idx < last_run_start_idx:
        return None

    # Parse start/end timestamps
    start_dt = parse_timestamp(lines[last_run_start_idx])
    end_dt = parse_timestamp(lines[last_train_idx])

    if start_dt is None or end_dt is None:
        return None

    duration_sec = (end_dt - start_dt).total_seconds()
    duration_min = duration_sec / 60.0

    # Extract iteration numbers
    first_iter_match = re.search(r"Iteration:\s*(\d+)", lines[last_run_start_idx])
    last_iter_match = re.search(r"Iteration:\s*(\d+)", lines[last_train_idx])
    first_iter = int(first_iter_match.group(1)) if first_iter_match else None
    last_iter = int(last_iter_match.group(1)) if last_iter_match else None

    return {
        "duration_min": round(duration_min, 2),
        "first_iter": first_iter,
        "last_iter": last_iter,
        "num_iters": (last_iter - first_iter + 1) if (first_iter is not None and last_iter is not None) else None,
        "start_dt": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_dt": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_cma_log(filepath):
    """
    Parse optimize_cma_log.log for the LAST CMA run.
    Uses the cumulative t[m:s] value from the last iteration line.
    Returns (duration_min, num_iterations, start_dt, end_dt) or None.
    """
    if not os.path.isfile(filepath):
        return None

    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Find the last CMA run ---
    # The "Iterat #Fevals" header line marks the start of each CMA run.
    last_header_idx = None
    for i, line in enumerate(lines):
        if "Iterat #Fevals" in line:
            last_header_idx = i

    if last_header_idx is None:
        return None

    # --- After the last header, find all CMA iteration lines ---
    # CMA iteration format: "INFO] iter_num  #fevals  f_value  axis_ratio  sigma  min_std  max_std  m:ss.s"
    cma_iter_pattern = re.compile(
        r"INFO\]\s+(\d+)\s+(\d+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+(\d+):(\d+\.?\d*)"
    )

    last_cma_line = None
    last_cma_idx = None
    for i in range(last_header_idx + 1, len(lines)):
        line = lines[i]
        if cma_iter_pattern.search(line):
            last_cma_line = line
            last_cma_idx = i

    if last_cma_line is None:
        return None

    # Parse the last CMA iteration line
    m = cma_iter_pattern.search(last_cma_line)
    if not m:
        return None

    iter_num = int(m.group(1))
    fevals = int(m.group(2))
    elapsed_min = int(m.group(8))
    elapsed_sec = float(m.group(9))
    duration_min = elapsed_min + elapsed_sec / 60.0

    # Parse timestamps
    start_dt = parse_timestamp(lines[last_header_idx])
    end_dt = parse_timestamp(last_cma_line)

    return {
        "duration_min": round(duration_min, 2),
        "num_iterations": iter_num,
        "num_fevals": fevals,
        "start_dt": start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else "",
        "end_dt": end_dt.strftime("%Y-%m-%d %H:%M:%S") if end_dt else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract training times from experiment logs to CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(BASE_DIR, "training_times.csv"),
        help="Output CSV file path (default: training_times.csv)"
    )
    args = parser.parse_args()

    # Discover all case directories (union of both experiments dirs)
    cases = set()
    for d in [EXPERIMENTS_DIR, EXPERIMENTS_OPT_DIR]:
        if os.path.isdir(d):
            for name in os.listdir(d):
                if os.path.isdir(os.path.join(d, name)):
                    cases.add(name)

    cases = sorted(cases)

    all_results = []

    for case_name in cases:
        print(f"Processing: {case_name}")

        inv_phy_path = os.path.join(EXPERIMENTS_DIR, case_name, "inv_phy_log.log")
        cma_path = os.path.join(EXPERIMENTS_OPT_DIR, case_name, "optimize_cma_log.log")

        inv_result = parse_inv_phy_log(inv_phy_path)
        cma_result = parse_cma_log(cma_path)

        row = {"case": case_name}

        if inv_result:
            row.update({
                "inv_phy_duration_min": inv_result["duration_min"],
                "inv_phy_first_iter": inv_result["first_iter"],
                "inv_phy_last_iter": inv_result["last_iter"],
                "inv_phy_num_iters": inv_result["num_iters"],
                "inv_phy_start": inv_result["start_dt"],
                "inv_phy_end": inv_result["end_dt"],
            })
            print(f"  inv_phy: {inv_result['duration_min']:.2f} min "
                  f"(iter {inv_result['first_iter']}→{inv_result['last_iter']})")
        else:
            print(f"  inv_phy: NOT FOUND or no data")

        if cma_result:
            row.update({
                "cma_duration_min": cma_result["duration_min"],
                "cma_iterations": cma_result["num_iterations"],
                "cma_fevals": cma_result["num_fevals"],
                "cma_start": cma_result["start_dt"],
                "cma_end": cma_result["end_dt"],
            })
            print(f"  cma:     {cma_result['duration_min']:.2f} min "
                  f"({cma_result['num_iterations']} iters, {cma_result['num_fevals']} fevals)")
        else:
            print(f"  cma:     NOT FOUND or no data")

        # Compute total
        inv_t = inv_result["duration_min"] if inv_result else 0
        cma_t = cma_result["duration_min"] if cma_result else 0
        row["total_duration_min"] = round(inv_t + cma_t, 2)
        if inv_result or cma_result:
            print(f"  TOTAL:   {row['total_duration_min']:.2f} min")

        all_results.append(row)

    if not all_results:
        print("No data found!")
        return

    # Determine all fieldnames (some rows may have missing keys)
    fieldnames = ["case"]
    # Fixed order for remaining columns
    remaining = [
        "cma_duration_min", "cma_iterations", "cma_fevals", "cma_start", "cma_end",
        "inv_phy_duration_min", "inv_phy_first_iter", "inv_phy_last_iter",
        "inv_phy_num_iters", "inv_phy_start", "inv_phy_end",
        "total_duration_min",
    ]
    fieldnames.extend(remaining)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDone! {len(all_results)} entries written to {args.output}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Case':<28} {'CMA(min)':>10} {'InvPhy(min)':>12} {'Total(min)':>11}")
    print("-" * 62)
    for row in all_results:
        cma = row.get("cma_duration_min", "")
        inv = row.get("inv_phy_duration_min", "")
        tot = row.get("total_duration_min", "")
        cma_s = f"{cma:.2f}" if isinstance(cma, (int, float)) else "N/A"
        inv_s = f"{inv:.2f}" if isinstance(inv, (int, float)) else "N/A"
        tot_s = f"{tot:.2f}" if isinstance(tot, (int, float)) else "N/A"
        print(f"{row['case']:<28} {cma_s:>10} {inv_s:>12} {tot_s:>11}")


if __name__ == "__main__":
    main()
