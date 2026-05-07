import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.pipeline.runtime_mixed_h2r import build_mixed_h2r_split, write_mixed_h2r_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _make_task(
    pair_root: Path,
    cache_root: Path,
    task: str,
    n: int,
    *,
    source_task: str | None = None,
    source_offset: int = 0,
) -> None:
    data_type = "h2r"
    duration = "1s"
    pair_dir = pair_root / data_type / duration / task
    cache_dir = cache_root / data_type / duration / task
    pair_rows = []
    cache_rows = []
    robot_task = source_task or task
    for i in range(n):
        pair_id = f"pair_{i:04d}"
        source_idx = source_offset + i
        clip_start = float(source_idx % 4)
        episode = f"ep{source_idx // 100:03d}"
        seg = f"seg{source_idx:04d}"
        robot_source_key = (
            f"{robot_task}/{episode}/{seg}_start{clip_start:.3f}_dur1.000"
        )
        base = {
            "clip_dur": 1.0,
            "clip_idx": source_idx % 4,
            "clip_start": clip_start,
            "data_type": data_type,
            "duration": duration,
            "episode": episode,
            "robot_source_key": robot_source_key,
            "robot_task": task,
            "seg": seg,
            "source_id": f"{robot_source_key}_{task}",
            "source_robot_task": robot_task,
            "source_segment_id": f"{robot_task}/{episode}/{seg}",
            "task": task,
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


def _append_uncached_pair_rows(pair_root: Path, task: str, start: int, n: int) -> None:
    data_type = "h2r"
    duration = "1s"
    pair_dir = pair_root / data_type / duration / task
    rows = [
        json.loads(line)
        for line in (pair_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for i in range(start, start + n):
        pair_id = f"pair_{i:04d}"
        rows.append({
            **rows[-1],
            "pair_id": pair_id,
            "source_id": f"{task}/uncached/{pair_id}",
            "source_segment_id": f"{task}/uncached/{pair_id}",
            "robot_source_key": f"{task}/uncached/{pair_id}",
            "video": f"video/{pair_id}.mp4",
            "control_video": f"control_video/{pair_id}.mp4",
        })
        (pair_dir / "video" / f"{pair_id}.mp4").parent.mkdir(parents=True, exist_ok=True)
        (pair_dir / "control_video" / f"{pair_id}.mp4").parent.mkdir(parents=True, exist_ok=True)
        (pair_dir / "video" / f"{pair_id}.mp4").write_bytes(b"")
        (pair_dir / "control_video" / f"{pair_id}.mp4").write_bytes(b"")
    _write_jsonl(pair_dir / "manifest.jsonl", rows)


def _args(tmp: Path, **overrides) -> Namespace:
    values = {
        "data_type": "h2r",
        "duration": "1s",
        "original_train_tasks": "Task_A,Task_B",
        "syn_train_tasks": "Task_A_syn,Task_B_syn",
        "ood_eval_tasks": "Task_C",
        "cache_root": str(tmp / "cache" / "vae"),
        "pair_root": str(tmp / "pair"),
        "original_train_size": 20,
        "syn_train_size": 30,
        "in_task_eval_size": 80,
        "ood_eval_size": 42,
        "data_seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


class MixedH2RSplitTest(unittest.TestCase):
    def test_stable_eval_tails_and_syn_train_only(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "Task_A", 100)
            _make_task(pair_root, cache_root, "Task_B", 100, source_offset=1000)
            _make_task(pair_root, cache_root, "Task_C", 50, source_offset=2000)
            _make_task(
                pair_root, cache_root, "Task_A_syn", 30,
                source_task="Task_A", source_offset=3000,
            )
            _make_task(
                pair_root, cache_root, "Task_B_syn", 30,
                source_task="Task_B", source_offset=4000,
            )

            split = build_mixed_h2r_split(_args(tmp))

            self.assertEqual(split.task_counts["original_train"], {"Task_A": 10, "Task_B": 10})
            self.assertEqual(split.task_counts["syn_train"], {"Task_A_syn": 15, "Task_B_syn": 15})
            self.assertEqual(split.task_counts["in_task_eval"], {"Task_A": 40, "Task_B": 40})
            self.assertEqual(split.task_counts["ood_eval"], {"Task_C": 42})
            self.assertTrue(all(r["mix_source"] == "syn" for r in split.syn_train_records))
            self.assertTrue(all(r["mix_source"] == "original" for r in split.eval_records))
            self.assertTrue(all(not r["robot_task"].endswith("_syn") for r in split.eval_records))
            self.assertTrue(all(not r["robot_task"].endswith("_syn") for r in split.ood_records))

            expected_eval = []
            for task in ("Task_A", "Task_B"):
                order_path = pair_root / "h2r" / "1s" / task / "pair_order.jsonl"
                ordered = [
                    json.loads(line)["pair_id"]
                    for line in order_path.read_text().splitlines()
                ]
                expected_eval.extend((task, pair_id) for pair_id in ordered[-40:])
            actual_eval = [(r["robot_task"], r["pair_id"]) for r in split.eval_records]
            self.assertEqual(actual_eval, expected_eval)

            eval_by_task = {}
            for task, pair_id in expected_eval:
                eval_by_task.setdefault(task, set()).add(pair_id)
            expected_original_train = []
            for task in ("Task_A", "Task_B"):
                selected = [
                    f"pair_{idx:04d}"
                    for idx in range(100)
                    if f"pair_{idx:04d}" not in eval_by_task[task]
                ][:10]
                expected_original_train.extend((task, pair_id) for pair_id in selected)
            actual_original_train = [
                (r["robot_task"], r["pair_id"])
                for r in split.original_train_records
            ]
            self.assertEqual(actual_original_train, expected_original_train)

            expected_syn_train = [
                ("Task_A_syn", f"pair_{idx:04d}") for idx in range(15)
            ] + [
                ("Task_B_syn", f"pair_{idx:04d}") for idx in range(15)
            ]
            actual_syn_train = [
                (r["robot_task"], r["pair_id"])
                for r in split.syn_train_records
            ]
            self.assertEqual(actual_syn_train, expected_syn_train)

            train_keys = {(r["robot_task"], r["pair_id"]) for r in split.train_records}
            eval_keys = {
                (r["robot_task"], r["pair_id"])
                for r in [*split.eval_records, *split.ood_records]
            }
            self.assertTrue(train_keys.isdisjoint(eval_keys))

    def test_robot_source_overlap_between_original_and_syn_fails(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "Task_A", 20)
            _make_task(pair_root, cache_root, "Task_A_syn", 20, source_task="Task_A")

            args = _args(
                tmp,
                original_train_tasks="Task_A",
                syn_train_tasks="Task_A_syn",
                ood_eval_tasks="",
                original_train_size=10,
                syn_train_size=10,
                in_task_eval_size=5,
                ood_eval_size=0,
            )
            with self.assertRaisesRegex(ValueError, "share robot_source_key"):
                build_mixed_h2r_split(args)

    def test_capacity_and_manifest_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "Task_A", 10)
            _make_task(pair_root, cache_root, "Task_A_syn", 5, source_offset=100)

            with self.assertRaisesRegex(ValueError, "Requested original train size"):
                build_mixed_h2r_split(_args(
                    tmp,
                    original_train_tasks="Task_A",
                    syn_train_tasks="Task_A_syn",
                    ood_eval_tasks="",
                    original_train_size=100,
                    syn_train_size=1,
                    in_task_eval_size=2,
                    ood_eval_size=0,
                ))

            (cache_root / "h2r" / "1s" / "Task_A_syn" / "pair_0000.pth").unlink()
            with self.assertRaises(FileNotFoundError):
                build_mixed_h2r_split(_args(
                    tmp,
                    original_train_tasks="Task_A",
                    syn_train_tasks="Task_A_syn",
                    ood_eval_tasks="",
                    original_train_size=1,
                    syn_train_size=1,
                    in_task_eval_size=2,
                    ood_eval_size=0,
                ))

    def test_syn_train_uses_available_cache_when_pair_manifest_has_uncached_tail(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "Task_A", 10)
            _make_task(pair_root, cache_root, "Task_A_syn", 5, source_offset=100)
            _append_uncached_pair_rows(pair_root, "Task_A_syn", start=5, n=3)
            args = _args(
                tmp,
                original_train_tasks="Task_A",
                syn_train_tasks="Task_A_syn",
                ood_eval_tasks="",
                original_train_size=4,
                syn_train_size=0,
                in_task_eval_size=2,
                ood_eval_size=0,
            )

            split = build_mixed_h2r_split(args)

            self.assertEqual(split.task_counts["syn_train"], {"Task_A_syn": 5})
            self.assertEqual(
                [record["pair_id"] for record in split.syn_train_records],
                [f"pair_{idx:04d}" for idx in range(5)],
            )

    def test_write_split_config_records_mixed_mode(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pair_root = tmp / "pair"
            cache_root = tmp / "cache" / "vae"
            _make_task(pair_root, cache_root, "Task_A", 10)
            _make_task(pair_root, cache_root, "Task_A_syn", 10, source_offset=100)
            args = _args(
                tmp,
                original_train_tasks="Task_A",
                syn_train_tasks="Task_A_syn",
                ood_eval_tasks="",
                original_train_size=4,
                syn_train_size=3,
                in_task_eval_size=2,
                ood_eval_size=0,
            )
            split = build_mixed_h2r_split(args)
            write_mixed_h2r_split(tmp / "run", args, split)

            config = json.loads((tmp / "run" / "data_split" / "config.json").read_text())
            self.assertEqual(config["mode"], "mixed_h2r")
            self.assertEqual(config["original_train_size"], 4)
            self.assertEqual(config["syn_train_size"], 3)
            self.assertEqual(config["actual_counts"]["syn_train"], {"Task_A_syn": 3})
            self.assertEqual(config["train_selection_order"], "pair_id_ascending")
            self.assertEqual(config["eval_selection_order"], "pair_order_tail")
            train_rows = [
                json.loads(line)
                for line in (tmp / "run" / "data_split" / "train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["mix_source"] for row in train_rows],
                ["original"] * 4 + ["syn"] * 3,
            )


if __name__ == "__main__":
    unittest.main()
