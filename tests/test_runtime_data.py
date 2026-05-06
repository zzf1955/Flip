import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.pipeline.runtime_data import (
    PAIR_ORDER_FILENAME,
    build_runtime_split,
    sample_eval_video_files,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _make_task(pair_root: Path, cache_root: Path, data_type: str,
               duration: str, task: str, n: int) -> None:
    pair_dir = pair_root / data_type / duration / task
    cache_dir = cache_root / data_type / duration / task
    pair_rows = []
    cache_rows = []
    for i in range(n):
        pair_id = f"pair_{i:04d}"
        source_id = f"{task}/ep000/seg{i:02d}_clip00"
        base = {
            "data_type": data_type,
            "duration": duration,
            "robot_task": task,
            "task": task,
            "source_id": source_id,
            "source_segment_id": f"{task}/ep000/seg{i:02d}",
        }
        pair_rows.append({
            **base,
            "video": f"video/{pair_id}.mp4",
            "control_video": f"control_video/{pair_id}.mp4",
        })
        cache_rows.append({**base, "cache_path": f"{pair_id}.pth"})
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{pair_id}.pth").write_bytes(b"")
    _write_jsonl(pair_dir / "manifest.jsonl", pair_rows)
    _write_jsonl(cache_dir / "manifest.jsonl", cache_rows)


def _args(tmp: Path, *, data_seed: int = 42) -> Namespace:
    return Namespace(
        data_type="h2r",
        duration="1s",
        train_tasks="Task_A,Task_B",
        ood_tasks="Task_C",
        cache_root=str(tmp / "cache" / "vae"),
        pair_root=str(tmp / "pair"),
        train_size=20,
        in_task_eval_size=8,
        in_task_video_size=4,
        ood_eval_size=3,
        ood_video_size=2,
        data_seed=data_seed,
    )


class RuntimeDataSplitTest(unittest.TestCase):
    def test_pair_order_split_is_reused_and_allocates_train_by_task_ratio(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_A", 10)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_B", 30)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_C", 5)

            split = build_runtime_split(_args(tmp, data_seed=42))

            self.assertEqual(split.split_counts["train"], {"Task_A": 5, "Task_B": 15})
            self.assertEqual(split.split_counts["in_task_eval"], {"Task_A": 2, "Task_B": 6})
            self.assertEqual(split.split_counts["ood_eval"], {"Task_C": 3})
            for task in ("Task_A", "Task_B", "Task_C"):
                self.assertTrue(
                    (pair_root / "h2r" / "1s" / task / PAIR_ORDER_FILENAME).is_file()
                )

            train_pairs = {
                (record["robot_task"], record["pair_id"])
                for record in split.train_records
            }
            eval_pairs = {
                (record["robot_task"], record["pair_id"])
                for record in split.eval_records
            }
            self.assertTrue(train_pairs.isdisjoint(eval_pairs))

            reused = build_runtime_split(_args(tmp, data_seed=999))
            self.assertEqual(split.train_files, reused.train_files)
            self.assertEqual(split.eval_files, reused.eval_files)
            self.assertEqual(split.ood_files, reused.ood_files)

    def test_eval_video_sampling_is_fixed_across_steps(self):
        files = [f"/tmp/pair_{i:04d}.pth" for i in range(12)]

        first = sample_eval_video_files(files, 4, data_seed=42, step=1, split_name="in_task")
        later = sample_eval_video_files(files, 4, data_seed=42, step=100, split_name="in_task")

        self.assertEqual(first, later)


if __name__ == "__main__":
    unittest.main()
