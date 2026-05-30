import unittest

import torch

import src.pipeline.eval_mitty.metrics as metrics


class EvalMittyMetricModelTest(unittest.TestCase):
    def test_patch_fid_with_no_fid_still_loads_inception_only(self):
        class DummyInception(torch.nn.Module):
            pass

        original_inception = metrics.InceptionFeatureExtractor
        metrics.InceptionFeatureExtractor = DummyInception
        try:
            lpips_model, inception, video_extractor = metrics.metric_models(
                torch.device("cpu"),
                no_lpips=True,
                no_fid=True,
                patch_fid=True,
                patch_fid_only=False,
            )
        finally:
            metrics.InceptionFeatureExtractor = original_inception

        self.assertIsNone(lpips_model)
        self.assertIsInstance(inception, DummyInception)
        self.assertIsNone(video_extractor)


if __name__ == "__main__":
    unittest.main()
