import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.pipeline.train_mitty import (
    detect_lora_config,
    resolve_lora_args,
    resolve_lora_target_modules,
)


def _add_lora_pair(sd, name: str, rank: int, in_dim: int = 8, out_dim: int = 12):
    sd[f"{name}.lora_A.default.weight"] = torch.zeros(rank, in_dim)
    sd[f"{name}.lora_B.default.weight"] = torch.zeros(out_dim, rank)


class LoraConfigTest(unittest.TestCase):
    def test_resolve_lora_target_modules_attention_controls(self):
        self.assertEqual(
            resolve_lora_target_modules(
                None,
                lora_attn_types="self",
                lora_attn_projections="q,o",
            ),
            ["self_attn.q", "self_attn.o"],
        )
        self.assertEqual(
            resolve_lora_target_modules(
                None,
                lora_attn_types="cross",
                lora_attn_projections="k,v",
            ),
            ["cross_attn.k", "cross_attn.v"],
        )

    def test_explicit_lora_target_modules_override_attention_controls(self):
        self.assertEqual(
            resolve_lora_target_modules(
                "ffn.0,ffn.2",
                lora_attn_types="self",
                lora_attn_projections="q",
            ),
            ["ffn.0", "ffn.2"],
        )

    def test_detect_lora_config_rank_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.safetensors"
            sd = {}
            for block in range(2):
                _add_lora_pair(sd, f"blocks.{block}.self_attn.q", rank=3)
                _add_lora_pair(sd, f"blocks.{block}.cross_attn.o", rank=3)
                _add_lora_pair(sd, f"blocks.{block}.ffn.0", rank=3)
            save_file(sd, path)

            info = detect_lora_config(str(path))

        self.assertEqual(info["rank"], 3)
        self.assertEqual(info["pairs"], 6)
        self.assertEqual(
            info["target_modules"],
            ["self_attn.q", "cross_attn.o", "ffn.0"],
        )

    def test_resolve_lora_args_autodetects_init_rank_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.safetensors"
            sd = {}
            _add_lora_pair(sd, "blocks.0.self_attn.k", rank=5)
            _add_lora_pair(sd, "blocks.0.cross_attn.v", rank=5)
            save_file(sd, path)

            args = Namespace(
                init_lora=str(path),
                lora_rank=None,
                lora_target_modules=None,
                lora_attn_types="self,cross",
                lora_attn_projections="q,k,v,o",
            )

            info = resolve_lora_args(args)

        self.assertEqual(info["rank"], 5)
        self.assertEqual(args.lora_rank, 5)
        self.assertEqual(args.lora_target_modules, "self_attn.k,cross_attn.v")


if __name__ == "__main__":
    unittest.main()
