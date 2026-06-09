"""Summarize robot_wam train-wan baseline and tune runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_name",
    "run_dir",
    "lora_rank",
    "lr",
    "action_loss_weight",
    "video_loss_weight",
    "state_tokens",
    "max_steps",
    "best_step",
    "best_metric",
    "best_metric_value",
    "best_eval_loss",
    "best_eval_video_loss",
    "best_eval_action_loss",
    "best_eval_in_task_loss",
    "best_eval_in_task_video_loss",
    "best_eval_in_task_action_loss",
    "best_eval_ood_loss",
    "best_eval_ood_video_loss",
    "best_eval_ood_action_loss",
    "best_eval_mean_loss",
    "final_train_step",
    "final_train_loss",
    "final_train_video_loss",
    "final_train_action_loss",
    "avg_step_time_s",
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_size_mib",
    "checkpoint_tensors",
    "checkpoint_parameters",
    "checkpoint_audit_ok",
    "checkpoint_forbidden_key_count",
    "checkpoint_unexpected_key_count",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(value)
    return rows


def token_to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value.replace("p", "."))


def parse_run_name(name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    rank = re.search(r"(?:^|_)r(\d+)(?:_|$)", name)
    lr = re.search(r"(?:^|_)lr([^_]+)(?:_|$)", name)
    action_weight = re.search(r"(?:^|_)aw([^_]+)(?:_|$)", name)
    steps = re.search(r"(?:^|_)s(\d+)k(?:_|$)", name)
    if rank is not None:
        parsed["lora_rank"] = int(rank.group(1))
    if lr is not None:
        parsed["lr"] = token_to_float(lr.group(1))
    if action_weight is not None:
        parsed["action_loss_weight"] = token_to_float(action_weight.group(1))
    if steps is not None:
        parsed["max_steps"] = int(steps.group(1)) * 1000
    return parsed


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    return f"{number:.{digits}g}"


def format_size(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    mib = size / 1024 / 1024
    return f"{mib:.1f} MiB", f"{mib:.3f}"


def latest_train_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if "train_loss" in row and "step" in row]
    if not train_rows:
        raise ValueError("train_log.jsonl has no train_loss rows")
    return max(train_rows, key=lambda row: int(row["step"]))


def average_step_time(rows: list[dict[str, Any]]) -> float:
    values = [float(row["time_s"]) for row in rows if "train_loss" in row and "time_s" in row]
    if not values:
        raise ValueError("train_log.jsonl has no train time_s rows")
    return sum(values) / len(values)


def best_eval_from_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [
        row for row in rows
        if ("eval_loss" in row or "eval_mean_loss" in row) and "step" in row
    ]
    if not eval_rows:
        raise ValueError("train_log.jsonl has no eval rows")
    return min(eval_rows, key=lambda row: float(row.get("eval_mean_loss", row.get("eval_loss"))))


def audit_checkpoint(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

    forbidden = [key for key in keys if "human" in key.lower() or "control" in key.lower()]
    unexpected = [
        key
        for key in keys
        if not (
            key.startswith("state_encoder.")
            or key.startswith("action_decoder.")
            or "lora_" in key.lower()
        )
    ]
    return {
        "checkpoint_tensors": len(keys),
        "checkpoint_audit_ok": len(keys) > 0 and not forbidden and not unexpected,
        "checkpoint_forbidden_key_count": len(forbidden),
        "checkpoint_unexpected_key_count": len(unexpected),
    }


def summarize_run(
    run_name: str,
    run_dir: Path,
    *,
    default_lr: float | None,
    default_state_tokens: int | None,
) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    log_rows = iter_jsonl(run_dir / "train_log.jsonl")
    best_summary = read_json(run_dir / "best_summary.json")
    train_summary = read_json(run_dir / "train_summary.json")

    parsed = parse_run_name(run_name)
    final_train = latest_train_row(log_rows)
    best_eval = best_summary.get("metrics")
    if not isinstance(best_eval, dict):
        best_eval = best_eval_from_log(log_rows)
    best_metric = best_summary.get("best_metric") or train_summary.get("best_metric")
    if not best_metric:
        best_metric = "eval_mean_loss" if "eval_mean_loss" in best_eval else "eval_loss"
    best_metric_value = best_summary.get("best_metric_value")
    if best_metric_value is None:
        best_metric_value = train_summary.get("best_metric_value")
    if best_metric_value is None and best_metric in best_eval:
        best_metric_value = best_eval[best_metric]

    checkpoint_path = best_summary.get("save", {}).get("path")
    checkpoint = Path(checkpoint_path) if checkpoint_path else run_dir / "best_checkpoint.safetensors"
    if not checkpoint.is_absolute():
        checkpoint = (run_dir / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    size_text, size_mib = format_size(checkpoint)
    audit = audit_checkpoint(checkpoint)

    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    state_action = config.get("state_action", {})

    lr = optimizer.get("lr", parsed.get("lr", default_lr))
    state_tokens = state_action.get("state_tokens", model.get("state_tokens", parsed.get("state_tokens", default_state_tokens)))
    max_steps = training.get("max_steps", parsed.get("max_steps", int(final_train["step"])))
    checkpoint_parameters = best_summary.get("save", {}).get("parameters")
    if checkpoint_parameters is None:
        checkpoint_parameters = train_summary.get("last_save", {}).get("parameters")

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "lora_rank": model.get("lora_rank", parsed.get("lora_rank", "")),
        "lr": format_float(lr),
        "action_loss_weight": format_float(loss.get("action_loss_weight", parsed.get("action_loss_weight"))),
        "video_loss_weight": format_float(loss.get("video_loss_weight")),
        "state_tokens": state_tokens if state_tokens is not None else "",
        "max_steps": max_steps,
        "best_step": int(best_eval["step"]),
        "best_metric": best_metric,
        "best_metric_value": format_float(best_metric_value),
        "best_eval_loss": format_float(best_eval.get("eval_loss", best_eval.get("eval_mean_loss"))),
        "best_eval_video_loss": format_float(best_eval.get("eval_video_loss")),
        "best_eval_action_loss": format_float(best_eval.get("eval_action_loss")),
        "best_eval_in_task_loss": format_float(best_eval.get("eval_in_task_loss")),
        "best_eval_in_task_video_loss": format_float(best_eval.get("eval_in_task_video_loss")),
        "best_eval_in_task_action_loss": format_float(best_eval.get("eval_in_task_action_loss")),
        "best_eval_ood_loss": format_float(best_eval.get("eval_ood_loss")),
        "best_eval_ood_video_loss": format_float(best_eval.get("eval_ood_video_loss")),
        "best_eval_ood_action_loss": format_float(best_eval.get("eval_ood_action_loss")),
        "best_eval_mean_loss": format_float(best_eval.get("eval_mean_loss")),
        "final_train_step": int(final_train["step"]),
        "final_train_loss": format_float(final_train["train_loss"]),
        "final_train_video_loss": format_float(final_train["train_video_loss"]),
        "final_train_action_loss": format_float(final_train["train_action_loss"]),
        "avg_step_time_s": format_float(average_step_time(log_rows)),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": size_text,
        "checkpoint_size_mib": size_mib,
        "checkpoint_tensors": audit["checkpoint_tensors"],
        "checkpoint_parameters": checkpoint_parameters if checkpoint_parameters is not None else "",
        "checkpoint_audit_ok": str(audit["checkpoint_audit_ok"]).lower(),
        "checkpoint_forbidden_key_count": audit["checkpoint_forbidden_key_count"],
        "checkpoint_unexpected_key_count": audit["checkpoint_unexpected_key_count"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_rows = [
        (row, float(row.get("best_metric_value") or row["best_eval_loss"]), float(row["action_loss_weight"]))
        for row in rows
        if (row.get("best_metric_value") or row.get("best_eval_loss")) and row.get("action_loss_weight")
    ]
    best_weighted = min(numeric_rows, key=lambda item: item[1])[0]
    same_weight = [item for item in numeric_rows if abs(item[2] - 1.0) < 1e-12]
    best_same_weight = min(same_weight, key=lambda item: item[1])[0] if same_weight else None
    audits_ok = all(row["checkpoint_audit_ok"] == "true" for row in rows)

    fields = [
        "run_name",
        "lr",
        "action_loss_weight",
        "max_steps",
        "best_step",
        "best_metric",
        "best_metric_value",
        "best_eval_loss",
        "best_eval_in_task_loss",
        "best_eval_ood_loss",
        "best_eval_mean_loss",
        "best_eval_video_loss",
        "best_eval_action_loss",
        "final_train_loss",
        "avg_step_time_s",
        "checkpoint_size",
        "checkpoint_audit_ok",
    ]
    lines = [
        "# Robot WAM train-wan sweep summary",
        "",
        md_table(rows, fields),
        "",
        "## Recommendation",
        "",
        f"- Best weighted eval loss: `{best_weighted['run_name']}` "
        f"at step {best_weighted['best_step']} with "
        f"{best_weighted['best_metric']}={best_weighted['best_metric_value']}.",
    ]
    if best_same_weight is not None:
        lines.append(
            f"- Best action_loss_weight=1.0 run: `{best_same_weight['run_name']}` "
            f"at step {best_same_weight['best_step']} with "
            f"{best_same_weight['best_metric']}={best_same_weight['best_metric_value']}."
        )
    lines.append(f"- Checkpoint key audit: {'PASS' if audits_ok else 'FAIL'}.")
    path.write_text("\n".join(lines) + "\n")


def collect_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    if args.baseline_dir:
        runs.append((args.baseline_name, Path(args.baseline_dir).resolve()))
    if args.tune_dir:
        tune_dir = Path(args.tune_dir).resolve()
        for child in sorted(tune_dir.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                runs.append((child.name, child))
    for item in args.run:
        name, sep, path = item.partition("=")
        if not sep:
            run_path = Path(name).resolve()
            runs.append((run_path.name, run_path))
        else:
            runs.append((name, Path(path).resolve()))
    if not runs:
        raise ValueError("no runs selected")
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", help="baseline run directory")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--baseline-lr", type=float, default=None)
    parser.add_argument("--tune-dir", help="directory containing tune run subdirectories")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="extra run as PATH or NAME=PATH; can be repeated",
    )
    parser.add_argument("--default-state-tokens", type=int, default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--require-audit", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        summarize_run(
            name,
            path,
            default_lr=args.baseline_lr if name == args.baseline_name else None,
            default_state_tokens=args.default_state_tokens,
        )
        for name, path in collect_runs(args)
    ]
    if args.require_audit:
        failed = [row["run_name"] for row in rows if row["checkpoint_audit_ok"] != "true"]
        if failed:
            raise SystemExit(f"checkpoint audit failed: {', '.join(failed)}")
    write_csv(Path(args.out_csv).resolve(), rows)
    write_markdown(Path(args.out_md).resolve(), rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
