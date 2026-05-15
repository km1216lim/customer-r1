"""GRPO entry point — verl-based, topology-aware.

Customer-R1 GRPO:
  - Actor + reference model initialized from the SFT checkpoint
  - vLLM rollout, either collocated (time-sliced) or disaggregated
  - Verifiable action-correctness reward (train/reward.py)
  - DeepSpeed-Ulysses SP=4 on Qwen2.5 (matches num_kv_heads)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--base_config", type=Path, default=Path("configs/grpo_base.yaml"))
    ap.add_argument("--output_dir", type=Path, default=Path("ckpt/grpo"))
    args = ap.parse_args()

    from topology import load_topology
    topo = load_topology(args.topology_config, args.topology)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))

    from verl.trainer.main_ppo import RayPPOTrainer
    from verl.utils.config import OmegaConfig
    from reward import verl_reward_fn

    rollout_share_gpu = (topo.rollout.mode == "collocated")
    rollout_n_gpus = topo.rollout.rollout_gpus or topo.world_size

    cfg = OmegaConfig.create({
        "algorithm": {
            "adv_estimator": "grpo",
            "kl_coef": base["grpo"]["kl_coef"],
            "clip_range": base["grpo"]["clip_range"],
            "group_size": topo.rollout.n_samples,
            "loss_type": base["grpo"]["loss_type"],
            "reward_normalization": base["grpo"]["reward_normalization"],
        },
        "data": {
            "train_files": base["data"]["train_path"],
            "val_files":   base["data"]["val_path"],
            "max_prompt_length":   topo.context_length - topo.completion_length,
            "max_response_length": topo.completion_length,
            "prompt_key":          "prompt_ids",
            "extra_info_key":      "action_gt",
            "pre_tokenized":       True,
            "train_batch_size":    topo.effective_batch_size,
        },
        "actor_rollout_ref": {
            "model": {
                "path": base["actor"]["init_from_ckpt"],
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "use_flash_attention_2": True,
            },
            "actor": {
                "optim": {
                    "lr": base["actor"]["lr"],
                    "weight_decay": base["actor"]["weight_decay"],
                    "warmup_steps": base["actor"]["warmup_steps"],
                    "lr_scheduler": base["actor"]["lr_scheduler"],
                    "max_grad_norm": base["actor"]["max_grad_norm"],
                },
                "ppo_micro_batch_size_per_gpu": topo.per_device_micro_batch,
                "ulysses_sequence_parallel_size": topo.sp_size,
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            "ref": {
                "model": {"path": base["ref_model"]["init_from_ckpt"]},
                "ulysses_sequence_parallel_size": topo.sp_size,
            },
            "rollout": {
                "name": "vllm",
                "tensor_model_parallel_size": topo.rollout.tp_size,
                "gpu_memory_utilization": 0.55 if rollout_share_gpu else 0.85,
                "n": topo.rollout.n_samples,
                "temperature": base["grpo"]["rollout_temperature"],
                "top_p": base["grpo"]["rollout_top_p"],
                "max_num_seqs": 64,
                "enable_prefix_caching": True,
                "share_gpu_with_actor": rollout_share_gpu,
                "n_gpus_per_node": rollout_n_gpus // max(1, (rollout_n_gpus + 7) // 8),
            },
        },
        "trainer": {
            "default_local_dir": str(args.output_dir),
            "total_training_steps": base["train"]["total_steps"],
            "save_freq": base["train"]["save_every_n_steps"],
            "test_freq": base["train"]["eval_every_n_steps"],
            "logger": "wandb" if base["logging"]["use_wandb"] else "console",
            "project_name": base["logging"]["project"],
            "experiment_name": f"{base['logging']['run_name_prefix']}-{topo.key}",
            "seed": base["train"]["seed"],
            "n_gpus_per_node": min(8, topo.world_size),
            "nnodes": (topo.world_size + 7) // 8,
        },
    })

    trainer = RayPPOTrainer(cfg, reward_fn=verl_reward_fn)
    trainer.fit()


if __name__ == "__main__":
    main()
