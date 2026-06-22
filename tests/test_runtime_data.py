import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.pipeline.runtime_data import (
    PAIR_ORDER_FILENAME,
    build_runtime_split,
    build_tail_eval_split,
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


def _explicit_split_row(
    data_type: str,
    duration: str,
    task: str,
    pair_idx: int,
    split: str,
    *,
    eval_role: str = "",
    domain: str = "id",
) -> dict:
    pair_id = f"pair_{pair_idx:04d}"
    return {
        "data_type": data_type,
        "duration": duration,
        "robot_task": task,
        "task": task,
        "pair_id": pair_id,
        "split": split,
        "eval_role": eval_role,
        "domain": domain,
        "video": f"{task}/video/{pair_id}.mp4",
        "control_video": f"{task}/control_video/{pair_id}.mp4",
        "source_id": f"{task}/explicit/{pair_id}",
        "source_segment_id": f"{task}/explicit/{pair_id}",
    }


def _explicit_args(tmp: Path, *, train_size: int = 7) -> Namespace:
    return Namespace(
        data_type="h2r",
        duration="1s",
        train_tasks="",
        ood_tasks="",
        cache_root=str(tmp / "cache" / "vae"),
        pair_root=str(tmp / "pair"),
        split_source="explicit",
        split_root=str(tmp / "pair" / "h2r" / "1s"),
        train_size=train_size,
        in_task_eval_size=2,
        in_task_video_size=2,
        ood_eval_size=2,
        ood_video_size=2,
        data_seed=42,
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

    def test_explicit_split_reads_root_jsonl_and_balances_train_by_task(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_A", 7)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_B", 5)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_C", 4)

            split_root = pair_root / "h2r" / "1s"
            _write_jsonl(
                split_root / "train.jsonl",
                [
                    *[
                        _explicit_split_row("h2r", "1s", "Task_A", idx, "train")
                        for idx in range(5)
                    ],
                    *[
                        _explicit_split_row("h2r", "1s", "Task_B", idx, "train")
                        for idx in range(3)
                    ],
                ],
            )
            _write_jsonl(
                split_root / "eval.jsonl",
                [
                    _explicit_split_row(
                        "h2r", "1s", "Task_A", 5, "eval",
                        eval_role="in_task", domain="id",
                    ),
                    _explicit_split_row(
                        "h2r", "1s", "Task_B", 3, "eval",
                        eval_role="in_task", domain="id",
                    ),
                    _explicit_split_row(
                        "h2r", "1s", "Task_C", 0, "eval",
                        eval_role="ood", domain="ood",
                    ),
                    _explicit_split_row(
                        "h2r", "1s", "Task_C", 1, "eval",
                        eval_role="ood", domain="ood",
                    ),
                ],
            )
            _write_jsonl(
                split_root / "test.jsonl",
                [
                    _explicit_split_row("h2r", "1s", "Task_A", 6, "test"),
                    _explicit_split_row("h2r", "1s", "Task_B", 4, "test"),
                    _explicit_split_row("h2r", "1s", "Task_C", 2, "test", domain="ood"),
                    _explicit_split_row("h2r", "1s", "Task_C", 3, "test", domain="ood"),
                ],
            )

            split = build_runtime_split(_explicit_args(tmp, train_size=7))

            self.assertEqual(split.split_source, "explicit")
            self.assertEqual(split.split_counts["train"], {"Task_A": 4, "Task_B": 3})
            self.assertEqual(split.split_counts["in_task_eval"], {"Task_A": 1, "Task_B": 1})
            self.assertEqual(split.split_counts["ood_eval"], {"Task_C": 2})
            self.assertEqual(split.split_counts["test"], {"Task_A": 1, "Task_B": 1, "Task_C": 2})
            self.assertEqual(len(split.train_files), 7)
            self.assertEqual(len(split.eval_files), 2)
            self.assertEqual(len(split.ood_files), 2)
            self.assertEqual(len(split.test_files), 4)

            self.assertTrue(all(record["split"] == "train" for record in split.train_records))
            self.assertTrue(all(record["split"] == "eval" for record in split.eval_records))
            self.assertTrue(all(record["eval_role"] == "in_task" for record in split.eval_records))
            self.assertTrue(all(record["eval_role"] == "ood" for record in split.ood_records))
            self.assertTrue(all(Path(path).is_file() for path in split.train_files))

    def test_eval_video_sampling_is_fixed_across_steps(self):
        files = [f"/tmp/pair_{i:04d}.pth" for i in range(12)]

        first = sample_eval_video_files(files, 4, data_seed=42, step=1, split_name="in_task")
        later = sample_eval_video_files(files, 4, data_seed=42, step=100, split_name="in_task")

        self.assertEqual(first, later)

    def test_tail_eval_split_reads_percent_from_pair_order_tail(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_A", 10)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_B", 20)
            _make_task(pair_root, cache_root, "h2r", "1s", "Task_C", 5)

            args = _args(tmp, data_seed=42)
            args.eval_tail_percent = 20.0
            split = build_tail_eval_split(args)

            expected_eval = []
            for task in ("Task_A", "Task_B"):
                order_path = pair_root / "h2r" / "1s" / task / PAIR_ORDER_FILENAME
                ordered = [
                    json.loads(line)["pair_id"]
                    for line in order_path.read_text().splitlines()
                ]
                expected_eval.extend(
                    (task, pair_id)
                    for pair_id in ordered[-max(1, len(ordered) // 5):]
                )

            expected_ood = []
            order_path = pair_root / "h2r" / "1s" / "Task_C" / PAIR_ORDER_FILENAME
            ordered = [
                json.loads(line)["pair_id"]
                for line in order_path.read_text().splitlines()
            ]
            expected_ood.extend(("Task_C", pair_id) for pair_id in ordered[-1:])

            actual_eval = [
                (record["robot_task"], record["pair_id"])
                for record in split.eval_records
            ]
            actual_ood = [
                (record["robot_task"], record["pair_id"])
                for record in split.ood_records
            ]

            self.assertEqual(actual_eval, expected_eval)
            self.assertEqual(actual_ood, expected_ood)
            self.assertEqual(split.split_counts["in_task_eval"], {"Task_A": 2, "Task_B": 4})
            self.assertEqual(split.split_counts["ood_eval"], {"Task_C": 1})


if __name__ == "__main__":
    unittest.main()
