"""SFT entry point — verl 0.4.1 (FSDP2), topology-aware.

verl 0.4.1's `verl.trainer.fsdp_sft_trainer` exposes `run_sft(config)`; its
hydra entry is a thin wrapper:

    @hydra.main(config_path="config", config_name="sft_trainer", ...)
    def main(config): run_sft(config)

Rather than re-declaring verl's full config schema (which drifts across
versions), we load verl's *bundled* `config/sft_trainer.yaml` as the base
so all version-specific defaults are present, override only the fields
Customer-R1 needs, then call `run_sft(cfg)` directly under torchrun.

Data: data/processed*/{train,test}.parquet from tokenize_pack_compressed.py.
  prompt_key = "prompt_text"      (chat-templated, ends at the assistant
                                   generation prompt)
  response_key = "completion_text" (single-JSON paper format)
verl's single-turn SFTDataset tokenizes prompt + response and masks the
prompt span, computing loss on the response only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--base_config", type=Path, default=Path("configs/sft_base.yaml"))
    ap.add_argument("--output_dir", type=Path, default=None)
    args = ap.parse_args()

    # Allow `python train/sft.py` and `torchrun train/sft.py` to import topology.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from topology import load_topology

    topo = load_topology(args.topology_config, args.topology)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))

    from omegaconf import OmegaConf
    import verl.trainer.fsdp_sft_trainer as sft_mod
    from verl.trainer.fsdp_sft_trainer import run_sft

    # Load verl's bundled base config so version-specific defaults stay intact.
    verl_yaml = Path(sft_mod.__file__).parent / "config" / "sft_trainer.yaml"
    cfg = OmegaConf.load(str(verl_yaml))
    # Allow overriding keys that may not exist in older bundled yamls.
    OmegaConf.set_struct(cfg, False)

    # --- data --------------------------------------------------------------
    cfg.data.train_files = base["data"]["train_path"]
    cfg.data.val_files = base["data"]["val_path"]
    cfg.data.prompt_key = "prompt_text"
    cfg.data.response_key = "completion_text"
    cfg.data.max_length = int(topo.context_length)
    cfg.data.train_batch_size = int(base["train"]["batch_size"])
    cfg.data.micro_batch_size_per_gpu = int(topo.per_device_micro_batch)

    # --- model -------------------------------------------------------------
    cfg.model.partial_pretrain = topo.model_name
    cfg.model.enable_gradient_checkpointing = True

    # --- top-level parallelism / padding ----------------------------------
    cfg.ulysses_sequence_parallel_size = int(topo.sp_size)
    cfg.use_remove_padding = True

    # --- optim (paper §4.3: lr 1e-5, warmup 150 steps, cosine) -------------
    total_steps = int(base["train"]["total_steps"])
    cfg.optim.lr = float(base["optim"]["lr"])
    cfg.optim.weight_decay = float(base["optim"]["weight_decay"])
    cfg.optim.lr_scheduler = base["optim"]["lr_scheduler"]
    cfg.optim.clip_grad = float(base["optim"]["max_grad_norm"])
    # verl uses a warmup *ratio* (fraction of total steps); the paper gives an
    # absolute step count, so convert.
    cfg.optim.warmup_steps_ratio = float(base["optim"]["warmup_steps"]) / total_steps

    # --- trainer -----------------------------------------------------------
    nnodes = int(os.environ.get("NNODES", "1"))
    cfg.trainer.total_training_steps = total_steps
    cfg.trainer.project_name = base["logging"]["project"]
    cfg.trainer.experiment_name = f"{base['logging']['run_name_prefix']}-{topo.key}"
    out_dir = str(args.output_dir) if args.output_dir else f"ckpt/{base['logging']['run_name_prefix']}"
    cfg.trainer.default_local_dir = out_dir
    cfg.trainer.logger = ["console", "wandb"] if base["logging"].get("use_wandb") else ["console"]
    cfg.trainer.save_freq = int(base["train"]["save_every_n_steps"])
    cfg.trainer.test_freq = int(base["train"]["eval_every_n_steps"])
    cfg.trainer.seed = int(base["train"]["seed"])
    cfg.trainer.nnodes = nnodes
    cfg.trainer.n_gpus_per_node = int(topo.world_size) // nnodes

    if int(os.environ.get("RANK", "0")) == 0:
        print("[sft] resolved config:")
        print(OmegaConf.to_yaml(cfg))

    run_sft(cfg)


if __name__ == "__main__":
    main()
