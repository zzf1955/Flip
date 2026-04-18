"""Convert train.log produced by train_mitty / train_lora to a train.csv.

Useful for runs that were launched before the live-CSV feature was added.
The script is idempotent and can be run while training is still going — it
just snapshots the log file at invocation time.

Usage:
    python -m src.tools.train_log_to_csv training_data/log/<RUN>/train.log
    # Writes training_data/log/<RUN>/train.csv

    # Or watch/regenerate repeatedly with --watch N seconds
    python -m src.tools.train_log_to_csv training_data/log/<RUN>/train.log --watch 10
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path


STEP_RE = re.compile(
    r"step=(\d+)/\d+ train_loss=([\d.eE+-]+) lr=([\d.eE+-]+) time=([\d.]+)s")
EVAL_IN_TASK_RE = re.compile(r"EVAL eval_loss_in_task=([\d.eE+-]+)")
EVAL_OOD_RE = re.compile(r"EVAL eval_loss_ood=([\d.eE+-]+)")
SAVE_RE = re.compile(r"SAVE .*?(step-\d+\.safetensors)")
VIDEO_RE = re.compile(r"EVAL VIDEO.*?(step-\d+)")


HEADERS = [
    "step", "train_loss", "lr", "time_s",
    "eval_loss_in_task", "eval_loss_ood",
    "save_ckpt", "eval_video",
]


def parse_log(log_path: Path) -> list[dict]:
    """Walk train.log line by line, emitting one dict per training step.

    Eval and save markers follow the step line they belong to, so we attribute
    them to the most recent step seen.
    """
    rows: dict[int, dict] = {}  # step -> row
    order: list[int] = []       # preserve insertion order
    current_step = None

    with open(log_path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {h: "" for h in HEADERS})
                rows[step].update({
                    "step": step,
                    "train_loss": m.group(2),
                    "lr": m.group(3),
                    "time_s": m.group(4),
                })
                if step not in order:
                    order.append(step)
                current_step = step
                continue

            if current_step is None:
                continue
            row = rows[current_step]

            m = EVAL_IN_TASK_RE.search(line)
            if m:
                row["eval_loss_in_task"] = m.group(1)
                continue

            m = EVAL_OOD_RE.search(line)
            if m:
                row["eval_loss_ood"] = m.group(1)
                continue

            m = SAVE_RE.search(line)
            if m:
                row["save_ckpt"] = m.group(1)
                continue

            m = VIDEO_RE.search(line)
            if m:
                row["eval_video"] = m.group(1)
                continue

    return [rows[s] for s in order]


def write_csv(rows: list[dict], csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Convert train.log → train.csv (loss / lr / eval per step)")
    ap.add_argument("log", help="path to train.log")
    ap.add_argument("--out", help="output csv path (default: sibling train.csv)")
    ap.add_argument("--watch", type=float, default=0,
                    help="regenerate every N seconds (0 = once)")
    args = ap.parse_args()

    log_path = Path(args.log).resolve()
    if not log_path.exists():
        print(f"Not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(args.out).resolve() if args.out else log_path.with_name("train.csv")

    while True:
        rows = parse_log(log_path)
        write_csv(rows, csv_path)
        print(f"{csv_path}: {len(rows)} rows")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
