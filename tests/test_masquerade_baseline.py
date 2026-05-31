from pathlib import Path
import unittest

from src.pipeline.masquerade_baseline import (
    PairRecord,
    _frame_index_for_output,
    select_records,
)


def _record(idx: int) -> PairRecord:
    return PairRecord(
        pair_id=f"pair_{idx:04d}",
        pair_index=idx,
        source_id=f"task/ep000/seg00_clip{idx:02d}",
        source_segment_id="task/ep000/seg00",
        task="task",
        episode="ep000",
        seg="seg00",
        clip_idx=idx,
        clip_start=0.0,
        clip_dur=1.0,
        augment="normal",
        human_src=Path("human.mp4"),
        robot_src=Path("robot.mp4"),
        control_video=Path("control.mp4"),
        robot_video=Path("target.mp4"),
        segment_joints=Path("seg_joints.parquet"),
        manifest={},
    )


class MasqueradeBaselineTest(unittest.TestCase):
    def test_frame_index_mapping_matches_4k_plus_1_endpoint_policy(self):
        self.assertEqual(_frame_index_for_output(0.0, 1.0, 0), 0)
        self.assertEqual(_frame_index_for_output(0.0, 1.0, 16), 29)
        self.assertEqual(_frame_index_for_output(3.0, 1.0, 16), 119)

    def test_select_records_filters_pair_ids_before_head(self):
        records = [_record(idx) for idx in range(5)]
        selected = select_records(records, head=1, pair_ids={"pair_0003", "pair_0004"})
        self.assertEqual([record.pair_id for record in selected], ["pair_0003"])

    def test_select_records_rejects_multiple_slice_modes(self):
        records = [_record(idx) for idx in range(5)]
        with self.assertRaisesRegex(ValueError, "Use only one"):
            select_records(records, head=1, tail=1)


if __name__ == "__main__":
    unittest.main()
