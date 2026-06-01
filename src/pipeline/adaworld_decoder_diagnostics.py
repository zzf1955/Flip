"""Diagnostics for AdaWorld latent action decoder prediction CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in --prediction {value!r}")
        path = Path(path_text).expanduser().resolve()
    else:
        path = Path(value).expanduser().resolve()
        label = path.parent.name or path.stem
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    return label, path


def discover_action_dims(df: pd.DataFrame) -> list[int]:
    dims: list[int] = []
    for column in df.columns:
        if column.startswith("action_target_"):
            dims.append(int(column.rsplit("_", 1)[1]))
    dims = sorted(dims)
    if not dims:
        raise ValueError("Prediction CSV contains no action_target_XX columns")
    for dim in dims:
        target_col = f"action_target_{dim:02d}"
        pred_col = f"action_pred_{dim:02d}"
        if target_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"Prediction CSV is missing {target_col} or {pred_col}")
    if dims != list(range(len(dims))):
        raise ValueError(f"Action dims must be contiguous from 0, got {dims}")
    return dims


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = math.sqrt(float(np.square(pred_centered).sum() * np.square(target_centered).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((pred_centered * target_centered).sum() / denom)


def summarize_prediction_csv(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dims = discover_action_dims(df)
    rows: list[dict[str, float | int | str]] = []
    for dim in dims:
        target = df[f"action_target_{dim:02d}"].to_numpy(dtype=np.float64)
        pred = df[f"action_pred_{dim:02d}"].to_numpy(dtype=np.float64)
        residual = pred - target
        mse = float(np.square(residual).mean())
        target_var = float(np.var(target))
        target_std = float(np.std(target))
        pred_std = float(np.std(pred))
        target_sse = float(np.square(target - target.mean()).sum())
        residual_sse = float(np.square(residual).sum())
        rows.append(
            {
                "split": label,
                "dim": dim,
                "n_samples": int(len(df)),
                "mse": mse,
                "rmse": float(math.sqrt(mse)),
                "mae": float(np.abs(residual).mean()),
                "bias": float(residual.mean()),
                "target_mean": float(target.mean()),
                "target_std": target_std,
                "pred_mean": float(pred.mean()),
                "pred_std": pred_std,
                "pred_std_ratio": float(pred_std / max(target_std, 1e-12)),
                "normalized_mse": float(mse / max(target_var, 1e-12)),
                "r2": float(1.0 - residual_sse / max(target_sse, 1e-12)),
                "corr": correlation(pred, target),
            }
        )
    return pd.DataFrame(rows)


def build_weights(
    summary: pd.DataFrame,
    *,
    source: str,
    metric: str,
    power: float,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    if power <= 0.0:
        raise ValueError(f"weight_power must be positive, got {power}")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError(f"Invalid weight bounds: min={min_weight} max={max_weight}")
    source_df = summary[summary["split"] == source].sort_values("dim")
    if source_df.empty:
        raise ValueError(f"weight_source={source!r} not found in summary splits")
    if metric == "normalized_mse":
        raw = source_df["normalized_mse"].to_numpy(dtype=np.float64)
    elif metric == "mse":
        raw = source_df["mse"].to_numpy(dtype=np.float64)
    elif metric == "one_minus_r2":
        raw = 1.0 - source_df["r2"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported weight_metric={metric!r}")
    if not np.isfinite(raw).all():
        raise ValueError("Weight metric contains non-finite values")
    raw = np.maximum(raw, 1e-12)
    weights = np.power(raw / raw.mean(), power)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
    return pd.DataFrame({"dim": source_df["dim"].to_numpy(dtype=np.int64), "weight": weights})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="prediction CSV, optionally label=path; can be passed multiple times",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-source", default="")
    parser.add_argument(
        "--weight-metric",
        choices=["normalized_mse", "mse", "one_minus_r2"],
        default="normalized_mse",
    )
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = [parse_prediction_arg(item) for item in args.prediction]
    summary = pd.concat(
        [summarize_prediction_csv(label, path) for label, path in parsed],
        ignore_index=True,
    )
    summary = summary.sort_values(["split", "mse"], ascending=[True, False])
    summary_path = out_dir / "per_dim_summary.csv"
    summary.to_csv(summary_path, index=False)

    weight_source = args.weight_source or parsed[0][0]
    weights = build_weights(
        summary,
        source=weight_source,
        metric=str(args.weight_metric),
        power=float(args.weight_power),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
    )
    weights_csv = out_dir / "loss_weights.csv"
    weights_json = out_dir / "loss_weights.json"
    weights.to_csv(weights_csv, index=False)
    weights_json.write_text(
        json.dumps(
            {
                "source": weight_source,
                "metric": args.weight_metric,
                "power": float(args.weight_power),
                "min_weight": float(args.min_weight),
                "max_weight": float(args.max_weight),
                "weights": [float(value) for value in weights["weight"].to_numpy()],
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "weights_csv": str(weights_csv),
                "weights_json": str(weights_json),
                "splits": [label for label, _ in parsed],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
