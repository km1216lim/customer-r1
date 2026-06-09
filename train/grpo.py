"""GRPO entry point — verl 0.4.1, topology-aware.

Mirrors the SFT approach in train/sft.py:
  - Load verl's bundled `config/ppo_trainer.yaml` as base so all version-specific
    defaults (critic schema, FSDP wrap policy, profiler, ...) are preserved.
  - Override only the fields Customer-R1 needs.
  - Wire our custom rule-based reward via `custom_reward_function.path / name`.
  - Call `verl.trainer.main_ppo.run_ppo(cfg)` which handles ray.init, tokenizer,
    role_worker_mapping, resource_pool_manager, dataset/trainer construction.

Reward weights from configs/grpo_{base,l2}.yaml are exported as environment
variables so the verl-loaded compute_score (train.reward.compute_score) can
read them without going through verl's hydra schema (which doesn't have a
slot for our domain-specific reward kwargs).

Data: we use a parquet enriched by data/enrich_for_rl.py — same prompt
content as SFT, but with the `data_source` and nested `reward_model.ground_truth`
columns verl's RLHFDataset expects.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _enriched_path(orig_path: str) -> str:
    """Map a SFT parquet path to the RL-enriched sibling produced by
    data/enrich_for_rl.py. data/processed_L2/train.parquet ->
    data/processed_L2/train_rl.parquet."""
    p = Path(orig_path)
    return str(p.with_name(p.stem + "_rl" + p.suffix))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--base_config", type=Path, default=Path("configs/grpo_base.yaml"))
    ap.add_argument("--output_dir", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from topology import load_topology
    topo = load_topology(args.topology_config, args.topology)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))

    # --- Export reward weights as env vars BEFORE verl loads compute_score.
    # train/reward.py:compute_score reads these on first call.
    for k, v in base["reward"].items():
        env_key = f"CUSTOMER_R1_{k.upper()}"
        os.environ[env_key] = str(v)

    from omegaconf import OmegaConf
    import verl.trainer.main_ppo as ppo_mod
    from verl.trainer.main_ppo import run_ppo

    # Load verl's bundled PPO config as base so all defaults are present.
    verl_yaml = Path(ppo_mod.__file__).parent / "config" / "ppo_trainer.yaml"
    cfg = OmegaConf.load(str(verl_yaml))
    OmegaConf.set_struct(cfg, False)

    nnodes = int(os.environ.get("NNODES", "1"))
    n_gpus_per_node = int(topo.world_size) // nnodes

    # --- data --------------------------------------------------------------
    # verl's RLHFDataset reads a `prompt` column (chat list or string) and a
    # nested `reward_model.ground_truth` column. Our SFT parquet has
    # prompt_text + action_gt; data/enrich_for_rl.py adds the missing columns.
    cfg.data.train_files = _enriched_path(base["data"]["train_path"])
    cfg.data.val_files = _enriched_path(base["data"]["val_path"])
    cfg.data.prompt_key = "prompt"
    cfg.data.reward_fn_key = "data_source"
    cfg.data.max_prompt_length = int(topo.context_length) - int(topo.completion_length)
    cfg.data.max_response_length = int(topo.completion_length)
    cfg.data.train_batch_size = int(base["train"]["batch_size"])
    cfg.data.return_raw_chat = True
    cfg.data.truncation = "left"  # left-truncate long prompts rather than error
    cfg.data.shuffle = True
    cfg.data.trust_remote_code = False

    # --- custom reward (rule-based) ----------------------------------------
    # train/reward.py:compute_score takes (data_source, solution_str,
    # ground_truth, extra_info=None) and returns a float per sample.
    cfg.custom_reward_function.path = str(Path(__file__).resolve().parent / "reward.py")
    cfg.custom_reward_function.name = "compute_score"
    cfg.reward_model.enable = False  # rule-based only, no learned reward model
    cfg.reward_model.reward_manager = "naive"

    # --- actor (policy) ----------------------------------------------------
    init_ckpt = base["actor"]["init_from_ckpt"]
    cfg.actor_rollout_ref.model.path = init_ckpt
    cfg.actor_rollout_ref.model.enable_gradient_checkpointing = True
    cfg.actor_rollout_ref.model.use_remove_padding = True
    cfg.actor_rollout_ref.model.trust_remote_code = False

    cfg.actor_rollout_ref.actor.strategy = "fsdp2"
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = int(base["train"]["batch_size"])
    cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = int(topo.per_device_micro_batch)
    cfg.actor_rollout_ref.actor.ulysses_sequence_parallel_size = int(topo.sp_size)
    cfg.actor_rollout_ref.actor.grad_clip = float(base["actor"]["max_grad_norm"])
    cfg.actor_rollout_ref.actor.clip_ratio = float(base["grpo"]["clip_range"])
    # GRPO uses KL as a loss term (not as a reward shaping signal).
    cfg.actor_rollout_ref.actor.use_kl_loss = True
    cfg.actor_rollout_ref.actor.kl_loss_coef = float(base["grpo"]["kl_coef"])
    cfg.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
    cfg.actor_rollout_ref.actor.entropy_coeff = 0.0
    cfg.actor_rollout_ref.actor.ppo_epochs = 1

    cfg.actor_rollout_ref.actor.optim.lr = float(base["actor"]["lr"])
    cfg.actor_rollout_ref.actor.optim.weight_decay = float(base["actor"]["weight_decay"])
    cfg.actor_rollout_ref.actor.optim.lr_warmup_steps = int(base["actor"]["warmup_steps"])
    cfg.actor_rollout_ref.actor.optim.warmup_style = "constant"

    # --- reference model ---------------------------------------------------
    cfg.actor_rollout_ref.ref.fsdp_config.param_offload = True  # 7B ref fits comfortably with offload
    # verl validates that AT LEAST ONE of log_prob_micro_batch_size /
    # log_prob_micro_batch_size_per_gpu is set; default null fails validation.
    cfg.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = int(topo.per_device_micro_batch)

    # --- rollout (vLLM) ----------------------------------------------------
    rollout_share_gpu = (topo.rollout.mode == "collocated")
    cfg.actor_rollout_ref.rollout.name = "vllm"
    cfg.actor_rollout_ref.rollout.mode = "sync"
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = int(topo.rollout.tp_size)
    cfg.actor_rollout_ref.rollout.n = int(topo.rollout.n_samples)
    cfg.actor_rollout_ref.rollout.temperature = float(base["grpo"]["rollout_temperature"])
    cfg.actor_rollout_ref.rollout.top_p = float(base["grpo"]["rollout_top_p"])
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.55 if rollout_share_gpu else 0.85
    cfg.actor_rollout_ref.rollout.max_model_len = int(topo.context_length)
    cfg.actor_rollout_ref.rollout.prompt_length = cfg.data.max_prompt_length
    cfg.actor_rollout_ref.rollout.response_length = cfg.data.max_response_length
    cfg.actor_rollout_ref.rollout.enforce_eager = True
    cfg.actor_rollout_ref.rollout.free_cache_engine = True
    cfg.actor_rollout_ref.rollout.enable_chunked_prefill = True
    cfg.actor_rollout_ref.rollout.dtype = "bfloat16"
    # vLLM 0.6.3 (our pin) doesn't know `disable_mm_preprocessor_cache` — that
    # was added in vLLM 0.7. verl's bundled ppo_trainer.yaml carries it under
    # engine_kwargs.vllm, and verl passes everything in that dict through to
    # EngineArgs, which then raises TypeError on the unknown kwarg. Drop the
    # field from the config so the multimodal-related extra is not forwarded.
    cfg.actor_rollout_ref.rollout.engine_kwargs.vllm = OmegaConf.create({"swap_space": None})
    # vLLM rejects (chunked_prefill=True AND max_num_batched_tokens < max_model_len).
    # The bundled default (8192) is too small for our 65K context — raise the
    # batched-token budget so chunked prefill's per-chunk window equals one full
    # prompt. This keeps chunked prefill's memory benefit (it splits prefill
    # *across* sequences in a batch) without forcing the prompt itself to be
    # cut into smaller pieces.
    cfg.actor_rollout_ref.rollout.max_num_batched_tokens = int(topo.context_length)
    # Same validation as ref above — rollout's log_prob batch needs a non-null value.
    cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = int(topo.per_device_micro_batch)

    # --- algorithm (GRPO) --------------------------------------------------
    cfg.algorithm.adv_estimator = "grpo"
    cfg.algorithm.norm_adv_by_std_in_grpo = (base["grpo"]["reward_normalization"] == "group_std")
    cfg.algorithm.use_kl_in_reward = False  # GRPO keeps KL inside the loss

    # --- critic ------------------------------------------------------------
    # GRPO doesn't use a critic, but verl's schema still validates the section.
    # Mirror the actor's strategy + model path so structural validation passes.
    # micro_batch_size_per_gpu needed for the same null-check as ref/rollout.
    cfg.critic.strategy = "fsdp2"
    cfg.critic.model.path = init_ckpt
    cfg.critic.ppo_micro_batch_size_per_gpu = int(topo.per_device_micro_batch)

    # --- trainer -----------------------------------------------------------
    out_dir = str(args.output_dir) if args.output_dir else f"ckpt/{base['logging']['run_name_prefix']}"
    cfg.trainer.default_local_dir = out_dir
    cfg.trainer.total_epochs = int(base["train"]["num_epochs"])
    cfg.trainer.save_freq = int(base["train"]["save_every_n_steps"])
    cfg.trainer.test_freq = int(base["train"]["eval_every_n_steps"])
    cfg.trainer.project_name = base["logging"]["project"]
    cfg.trainer.experiment_name = f"{base['logging']['run_name_prefix']}-{topo.key}"
    cfg.trainer.logger = ["console", "wandb"] if base["logging"].get("use_wandb") else ["console"]
    cfg.trainer.nnodes = nnodes
    cfg.trainer.n_gpus_per_node = n_gpus_per_node
    cfg.trainer.device = "cuda"
    cfg.trainer.val_before_train = False  # paper doesn't validate at step 0; saves ~5 min

    # --- ray ---------------------------------------------------------------
    # Leave num_cpus at the YAML default (null → Ray auto-detects).
    cfg.ray_init.num_cpus = None

    if int(os.environ.get("RANK", "0")) == 0:
        print("[grpo] resolved config:")
        print(OmegaConf.to_yaml(cfg))

    run_ppo(cfg)


if __name__ == "__main__":
    main()
