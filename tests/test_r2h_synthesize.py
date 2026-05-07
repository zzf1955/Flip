import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.pipeline.r2h_synthesize import (
    DEFAULT_EXCLUDE_EPISODES,
    build_manifest_record,
    collect_segment_clips,
    filter_excluded_clips,
    load_covered_robot_sources,
    select_clips,
    select_clips_proportional,
)


def _touch_segment(root: Path, task: str, episode: str, seg: str = "seg00") -> Path:
    path = root / task / episode / f"{seg}_video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-video")
    return path


class R2HSynthesizeTest(unittest.TestCase):
    def test_collects_segment_clips_with_stable_nonoverlap_keys(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            _touch_segment(root, "Task_A", "ep004")
            _touch_segment(root, "Task_A", "ep005")

            clips = collect_segment_clips(
                root, ["Task_A"], "1s", validate_videos=False)

            self.assertEqual(len(clips), 8)
            self.assertEqual(clips[0].robot_source_key,
                             "Task_A/ep004/seg00_start0.000_dur1.000")
            self.assertEqual(clips[1].robot_source_key,
                             "Task_A/ep004/seg00_start1.000_dur1.000")
            self.assertEqual(clips[-1].source_order_index, 7)

    def test_default_seedance_episodes_are_filtered_before_selection(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            _touch_segment(root, "Task_A", "ep000")
            _touch_segment(root, "Task_A", "ep004")
            clips = collect_segment_clips(
                root, ["Task_A"], "1s", validate_videos=False)

            eligible = filter_excluded_clips(
                clips, set(DEFAULT_EXCLUDE_EPISODES), set())

            self.assertEqual({clip.episode for clip in eligible}, {"ep004"})
            self.assertEqual(len(eligible), 4)

    def test_covered_manifest_filters_exact_robot_source_key(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            manifest = root / "covered.jsonl"
            covered_key = "Task_A/ep004/seg00_start1.000_dur1.000"
            manifest.write_text(json.dumps({"robot_source_key": covered_key}) + "\n")
            _touch_segment(root, "Task_A", "ep004")
            clips = collect_segment_clips(
                root, ["Task_A"], "1s", validate_videos=False)

            covered = load_covered_robot_sources([manifest])
            eligible = filter_excluded_clips(clips, set(), covered)

            self.assertNotIn(covered_key, {clip.robot_source_key for clip in eligible})
            self.assertEqual(len(eligible), 3)

    def test_select_clips_rejects_multiple_selection_modes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            _touch_segment(root, "Task_A", "ep004")
            clips = collect_segment_clips(
                root, ["Task_A"], "1s", validate_videos=False)

            with self.assertRaises(ValueError):
                select_clips(clips, num_samples=1, head=1)

    def test_proportional_selection_allocates_by_task_capacity(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            _touch_segment(root, "Task_A", "ep004")
            _touch_segment(root, "Task_B", "ep004")
            _touch_segment(root, "Task_B", "ep005")
            for i in range(4, 8):
                _touch_segment(root, "Task_C", f"ep{i:03d}")
            clips = collect_segment_clips(
                root, ["Task_A", "Task_B", "Task_C"], "1s",
                validate_videos=False)

            selected, counts = select_clips_proportional(clips, 7)

            self.assertEqual(counts, {"Task_A": 1, "Task_B": 2, "Task_C": 4})
            actual_counts = {}
            for clip in selected:
                actual_counts[clip.task] = actual_counts.get(clip.task, 0) + 1
            self.assertEqual(actual_counts, counts)

    def test_manifest_records_training_pair_roles_and_segment_source(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            _touch_segment(root, "Task_A", "ep004")
            clip = collect_segment_clips(
                root, ["Task_A"], "1s", validate_videos=False)[0]
            args = Namespace(
                duration="1s",
                cfg_scale=5.0,
                num_inference_steps=30,
            )

            record = build_manifest_record(
                clip=clip,
                pair_id="pair_0000",
                syn_task="Task_A_syn",
                rel_video="video/pair_0000.mp4",
                rel_control="control_video/pair_0000.mp4",
                args=args,
                run_name="Run",
                checkpoint_path=Path("ckpt/step-1000.safetensors"),
            )

            self.assertEqual(record["data_type"], "h2r")
            self.assertEqual(record["input_role"], "human")
            self.assertEqual(record["target_role"], "robot")
            self.assertEqual(record["robot_task"], "Task_A_syn")
            self.assertEqual(record["source_robot_task"], "Task_A")
            self.assertEqual(record["robot_source_key"], clip.robot_source_key)
            self.assertTrue(record["seedance_exclusion_checked"])


if __name__ == "__main__":
    unittest.main()
