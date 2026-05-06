import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.tools.eval_metrics import (
    clip_mask_indices,
    compute_pairwise_metrics,
    load_clip_mask_stack,
    mse_to_psnr,
    resolve_sam2_mask_path,
    split_video_by_mask,
)


class EvalMaskRegionTest(unittest.TestCase):
    def test_clip_mask_indices_match_16fps_clip_mapping(self):
        self.assertEqual(
            clip_mask_indices(clip_start=1.0, clip_dur=1.0, num_frames=17, mask_count=120),
            [30, 32, 34, 36, 38, 39, 41, 43, 45, 47, 49, 51, 52, 54, 56, 58, 59],
        )

    def test_load_clip_mask_stack_resolves_and_hflips_manifest_mask(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            mask_dir = root / "Task_A" / "ep000"
            mask_dir.mkdir(parents=True)
            masks = np.zeros((4, 2, 3), dtype=np.uint8)
            masks[0, :, 0] = 255
            masks[2, :, 1] = 255
            np.savez_compressed(mask_dir / "seg00.npz", masks=masks)

            record = {
                "robot_task": "Task_A",
                "episode": "ep000",
                "seg": "seg00",
                "clip_start": 0.0,
                "clip_dur": 0.1,
                "augment": "hflip",
            }

            self.assertEqual(
                resolve_sam2_mask_path(record, root),
                mask_dir / "seg00.npz",
            )
            clip_masks = load_clip_mask_stack(
                record,
                root,
                num_frames=2,
                frame_shape=(2, 3),
            )

            self.assertEqual(clip_masks.shape, (2, 2, 3))
            self.assertTrue(clip_masks[0, :, 2].all())
            self.assertTrue(clip_masks[1, :, 1].all())

    def test_split_video_by_mask_keeps_foreground_and_background_regions(self):
        frames = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3)
        masks = np.array(
            [
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ]
        )

        split = split_video_by_mask(frames, masks)

        self.assertTrue((split["foreground"][masks] == frames[masks]).all())
        self.assertTrue((split["foreground"][~masks] == 0).all())
        self.assertTrue((split["background"][~masks] == frames[~masks]).all())
        self.assertTrue((split["background"][masks] == 0).all())

    def test_pairwise_metrics_include_global_and_masked_regions(self):
        gt = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        gen = gt.copy()
        masks = np.zeros((1, 16, 16), dtype=bool)
        masks[:, :8, :] = True
        gen[masks] = 10

        metrics = compute_pairwise_metrics(
            gen,
            gt,
            lpips_model=None,
            device=torch.device("cpu"),
            masks=masks,
        )

        self.assertAlmostEqual(metrics["foreground_mse"], 100.0)
        self.assertAlmostEqual(metrics["background_mse"], 0.0)
        self.assertAlmostEqual(metrics["mse"], 50.0)
        self.assertAlmostEqual(metrics["foreground_psnr"], mse_to_psnr(100.0))
        self.assertEqual(metrics["background_psnr"], float("inf"))
        self.assertIn("foreground_ssim", metrics)
        self.assertIn("background_ssim", metrics)


if __name__ == "__main__":
    unittest.main()
