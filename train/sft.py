"""SFT entry point — verl-based, topology-aware.

Reads the topology config to set up:
  - DP / SP parallel groups (DeepSpeed-Ulysses)
  - Per-device micro batch and gradient accumulation
  - bf16 mixed precision, ZeRO-3, full activation checkpointing

Data:
  data/processed/sft_{train,val}.parquet  (from tokenize_pack.py)

Loss: standard next-token CE on completion_ids, masking the prompt portion.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--base_config", type=Path, default=Path("configs/sft_base.yaml"))
    ap.add_argument("--output_dir", type=Path, default=Path("ckpt/sft"))
    args = ap.parse_args()

    from topology import load_topology
    topo = load_topology(args.topology_config, args.topology)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))

    # Late import so `python sft.py --help` works without the heavy stack installed.
    from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
    from verl.utils.config import OmegaConfig

    # Build the OmegaConf the way verl's SFT trainer expects.
    cfg = OmegaConfig.create({
        "data": {
            "train_files": base["data"]["train_path"],
            "val_files":   base["data"]["val_path"],
            "max_length":  topo.context_length,
            "micro_batch_size_per_gpu": topo.per_device_micro_batch,
            "train_batch_size": topo.effective_batch_size,
            "prompt_key":     "prompt_ids",
            "response_key":   "completion_ids",
            "pre_tokenized":  True,
        },
        "model": {
            "partial_pretrain": topo.model_name,
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            "fsdp_config": {
                "wrap_policy": {"min_num_params": 0},
                "cpu_offload": False,
                "offload_params": False,
            },
            "ulysses_sequence_parallel_size": topo.sp_size,
            "use_flash_attention_2": True,
        },
        "optim": base["optim"],
        "trainer": {
            "default_local_dir": str(args.output_dir),
            "total_epochs": base["train"]["num_epochs"],
            "logger": "wandb" if base["logging"]["use_wandb"] else "console",
            "project_name": base["logging"]["project"],
            "experiment_name": f"{base['logging']['run_name_prefix']}-{topo.key}",
            "save_freq": base["train"]["save_every_n_steps"],
            "test_freq": base["train"]["eval_every_n_steps"],
            "seed": base["train"]["seed"],
        },
    })

    trainer = FSDPSFTTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
