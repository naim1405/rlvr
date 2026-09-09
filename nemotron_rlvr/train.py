"""GRPO training entrypoint for Qwen + NeMo Gym (P0–P6 `nemotron_verifier`)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
# Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. vLLM CuMem
# (sleep/wake colocated) asserts against it and dies before step 1.


def _schema_defaults() -> dict:
    """Super3 `setup()` indexes these with `cfg[key]`; missing → ConfigKeyError.

    Yaml values win (OmegaConf.merge defaults, yaml). Safe/off for 1×20GB.
    """
    return {
        "grpo": {
            "seed": 42,
            "use_dynamic_sampling": False,
            "batch_multiplier": 1,
            "dynamic_sampling_max_gen_batches": 10,
            "overlong_filtering": False,
            "val_at_end": False,
            "val_start_at": -1,
            "num_val_generations_per_prompt": 1,
            "val_num_generations_per_prompt": 1,
            "seq_logprob_error_threshold": None,
            "stop_at_validation_metric": None,
            "stop_at_validation_threshold": None,
            "reward_shaping": {
                "enabled": False,
                "overlong_buffer_length": 128,
                "overlong_buffer_penalty": 1,
                "max_response_length": "${policy.max_total_sequence_length}",
                "stop_properly_penalty_coef": None,
            },
            "reward_scaling": {
                "enabled": False,
                "source_min": 0.0,
                "source_max": 1.0,
                "target_min": 0.0,
                "target_max": 1.0,
            },
            "adv_estimator": {
                "name": "grpo",
                "minus_baseline": True,
                "normalize_rewards": "${grpo.normalize_rewards}",
                "use_leave_one_out_baseline": "${grpo.use_leave_one_out_baseline}",
            },
            "async_grpo": {
                "enabled": False,
                "max_trajectory_age_steps": 1,
                "in_flight_weight_updates": False,
                "recompute_kv_cache_after_weight_updates": False,
                "max_generation_failures": 0,
            },
        },
        "loss_fn": {
            "reference_policy_kl_type": "k3",
            "kl_input_clamp_value": 20.0,
            "kl_output_clamp_value": 10.0,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": False,
            "sequence_level_importance_ratios": False,
            "force_on_policy_ratio": False,
            "disable_ppo_ratio": False,
            "use_kl_in_reward": False,
            "truncated_importance_sampling_type": "tis",
            "truncated_importance_sampling_ratio_min": None,
            "positive_example_nll_weight": 0.0,
        },
        "data": {
            "shuffle": True,
            "num_workers": 1,
            "default": {
                "dataset_name": "NemoGymDataset",
                "env_name": "nemo_gym",
                "processor": "nemo_gym_data_processor",
            },
        },
        "logger": {
            "tensorboard": {},
            "swanlab": {"project": "rlvr", "name": "rlvr-single-gpu"},
            "mlflow": {
                "experiment_name": "rlvr",
                "run_name": "rlvr-single-gpu",
                "tracking_uri": "http://localhost:5000",
            },
            "gpu_monitoring": {
                "collection_interval": 10,
                "flush_interval": 10,
            },
        },
        "checkpointing": {
            "save_consolidated": False,
            "save_optimizer": True,
            "load_replay_buffer": False,
            "save_data_plane": False,
            "checkpoint_must_save_by": None,
        },
        "policy": {
            "offload_optimizer_for_logprob": False,
            "logprob_chunk_size": None,
            "hf_config_overrides": {},
            "max_grad_norm": 1.0,
            "optimizer": None,
            "scheduler": None,
            "dtensor_cfg": {"enabled": False},
            "dynamic_batching": {"enabled": False},
            "draft": {"enabled": False},
            "router_replay": {"enabled": False},
            "generation": {
                "top_k": None,
                "stop_token_ids": None,
                "stop_strings": None,
                "vllm_kwargs": {},
                "colocated": {
                    "enabled": True,
                    "resources": {"gpus_per_node": None, "num_nodes": None},
                },
                "vllm_cfg": {
                    "precision": "${policy.precision}",
                    "kv_cache_dtype": "auto",
                    "pipeline_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "hf_overrides": {},
                },
            },
            "megatron_cfg": {
                "empty_unused_memory_level": 2,
                "converter_type": "Qwen2ForCausalLM",
                "expert_tensor_parallel_size": 1,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "pipeline_dtype": "${policy.precision}",
                "sequence_parallel": False,
                "freeze_moe_router": True,
                "moe_router_dtype": "fp64",
                "moe_router_load_balancing_type": "none",
                "moe_router_bias_update_rate": 0.0,
                "moe_permute_fusion": False,
                "apply_rope_fusion": True,
                "bias_activation_fusion": True,
                "defer_fp32_logits": False,
                "moe_per_layer_logging": False,
                "moe_enable_deepep": False,
                "moe_token_dispatcher_type": "alltoall",
                "moe_shared_expert_overlap": False,
                "gradient_accumulation_fusion": False,
                "force_reconvert_from_hf": False,
                # Super3 gpt_model._postprocess calls process_mtp_loss when this
                # is not None (0 still counts). Logprob forwards pass labels=None
                # and crash. Qwen has no MTP — leave unset.
                "mtp_num_layers": None,
                "mtp_loss_scaling_factor": 0.0,
                "mtp_use_repeated_layer": False,
                "mtp_detach_heads": False,
                "cuda_graph_impl": "none",
                "cuda_graph_warmup_steps": 3,
                "use_gloo_process_groups": False,
                "recompute_granularity": "full",
                "fp8_cfg": None,
                "env_vars": None,
                "checkpoint": {
                    "async_save": False,
                    "ckpt_assume_constant_structure": True,
                },
                "peft": {"enabled": False},
                "optimizer": {
                    "min_lr": 3.0e-7,
                    "weight_decay": 0.01,
                    "bf16": True,
                    "fp16": False,
                    "params_dtype": "float32",
                    "adam_eps": 1.0e-8,
                    "sgd_momentum": 0.9,
                    "use_distributed_optimizer": False,
                    "use_precision_aware_optimizer": False,
                    "clip_grad": "${policy.max_grad_norm}",
                    "optimizer_cpu_offload": False,
                    "optimizer_offload_fraction": 0.0,
                },
                "scheduler": {
                    "start_weight_decay": 0.01,
                    "end_weight_decay": 0.01,
                    "weight_decay_incr_style": "constant",
                    "lr_decay_style": "constant",
                    "lr_decay_iters": "${grpo.max_num_steps}",
                    "lr_warmup_iters": 3,
                    "lr_warmup_init": 3.0e-7,
                },
                "distributed_data_parallel_config": {
                    "grad_reduce_in_fp32": False,
                    "overlap_grad_reduce": False,
                    "overlap_param_gather": False,
                    "use_custom_fsdp": False,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                },
            },
        },
    }


def _apply_schema_defaults(config):
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.create(_schema_defaults()), config)
    print("[train] merged Super3 schema defaults (yaml still wins)")
    return merged


def _maybe_register_omegaconf_resolvers() -> None:
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    if not OmegaConf.has_resolver("int_div"):
        OmegaConf.register_new_resolver("int_div", lambda x, y: int(x) // int(y))


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO on Qwen with NeMo Gym verifiers")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (relative to this file's directory or absolute)",
    )
    return parser.parse_args()


def _jsonl_path(data_cfg, split: str):
    from omegaconf import OmegaConf

    path = OmegaConf.select(data_cfg, f"{split}.data_path")
    if path:
        return path
    alt_key = f"{split}_jsonl_fpath"
    path = OmegaConf.select(data_cfg, alt_key)
    if path:
        return path
    if split == "validation":
        return OmegaConf.select(data_cfg, "val_jsonl_fpath")
    return None


def _should_use_gym(config) -> bool:
    try:
        from nemo_rl.algorithms.grpo import _should_use_nemo_gym as _fn
    except ImportError:
        from nemo_rl.algorithms.grpo import should_use_nemo_gym as _fn
    return bool(_fn(config))


def _configure_val_sizes(config, val_dataset) -> None:
    """Honor yaml caps. Never silently set val_batch_size = len(val) on 20GB."""
    from omegaconf import OmegaConf, open_dict

    n = len(val_dataset)
    configured = OmegaConf.select(config, "grpo.max_val_samples")
    with open_dict(config):
        if configured in (None, "null"):
            # Official Gym runner uses the full val set. Fine on multi-GPU;
            # dangerous on 1×20GB unless the yaml already capped it.
            config.grpo.max_val_samples = n
            if OmegaConf.select(config, "grpo.val_batch_size") in (None, "null"):
                config.grpo.val_batch_size = n
            return
        cap = min(int(configured), n)
        config.grpo.max_val_samples = cap
        vbs = OmegaConf.select(config, "grpo.val_batch_size")
        if vbs in (None, "null"):
            config.grpo.val_batch_size = cap
        else:
            config.grpo.val_batch_size = min(int(vbs), cap)


def _unpack_setup(result):
    """v0.5.0 setup() is a 10-tuple. Super3 returns 11 with Gym at index 2.

    Super3 NemoGym exposes `_spinup`, not `health_check`, so do not require
    health_check to treat index 2 as the Gym actor.
    """
    result = tuple(result)
    print(f"[train] setup() returned {len(result)} values: {[type(x).__name__ for x in result]}")

    if len(result) == 10:
        (
            policy,
            policy_generation,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        ) = result
        return {
            "policy": policy,
            "policy_generation": policy_generation,
            "nemo_gym": None,
            "cluster": cluster,
            "dataloader": dataloader,
            "val_dataloader": val_dataloader,
            "loss_fn": loss_fn,
            "logger": logger,
            "checkpointer": checkpointer,
            "grpo_state": grpo_state,
            "master_config": master_config,
        }
    if len(result) >= 11:
        (
            policy,
            policy_generation,
            nemo_gym,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            *_,
        ) = result
        if isinstance(nemo_gym, (tuple, list)):
            raise RuntimeError(
                f"setup() 11-tuple index 2 looks like a cluster ({type(nemo_gym)!r}), "
                "not a Gym actor. Cannot unpack."
            )
        return {
            "policy": policy,
            "policy_generation": policy_generation,
            "nemo_gym": nemo_gym,
            "cluster": cluster,
            "dataloader": dataloader,
            "val_dataloader": val_dataloader,
            "loss_fn": loss_fn,
            "logger": logger,
            "checkpointer": checkpointer,
            "grpo_state": grpo_state,
            "master_config": master_config,
        }
    raise RuntimeError(
        f"nemo_rl.algorithms.grpo.setup returned {len(result)} values; "
        "expected 10 (v0.5.0) or ≥11 (Super3)."
    )


def _call_grpo_train(grpo_train, packed, tokenizer, task_to_env):
    """Bind by name so v0.5 and Super3 signatures both work."""
    import inspect

    sig = inspect.signature(grpo_train)
    available = {
        "policy": packed["policy"],
        "policy_generation": packed["policy_generation"],
        "dataloader": packed["dataloader"],
        "val_dataloader": packed["val_dataloader"],
        "tokenizer": tokenizer,
        "loss_fn": packed["loss_fn"],
        "task_to_env": task_to_env,
        "val_task_to_env": task_to_env,
        "logger": packed["logger"],
        "checkpointer": packed["checkpointer"],
        "grpo_save_state": packed["grpo_state"],
        "grpo_state": packed["grpo_state"],
        "master_config": packed["master_config"],
    }
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"grpo_train requires {name!r} which train.py does not supply. "
                f"signature={list(sig.parameters)}"
            )
    print(f"[train] calling grpo_train({', '.join(kwargs)})")
    return grpo_train(**kwargs)


def _shutdown_cluster(cluster) -> None:
    clusters = cluster if isinstance(cluster, (tuple, list)) else (cluster,)
    for item in clusters:
        if item is None:
            continue
        shutdown = getattr(item, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[train] cluster shutdown: {exc}")


def _ensure_generation_defaults(generation_cfg):
    """OmegaConf raises on missing keys; v0.5 configure_generation_config indexes them."""
    from omegaconf import open_dict

    with open_dict(generation_cfg):
        if "top_k" not in generation_cfg:
            generation_cfg.top_k = None
        if "stop_token_ids" not in generation_cfg:
            generation_cfg.stop_token_ids = None
        if "stop_strings" not in generation_cfg:
            generation_cfg.stop_strings = None
        if "vllm_kwargs" not in generation_cfg:
            generation_cfg.vllm_kwargs = {}
        if "colocated" not in generation_cfg:
            generation_cfg.colocated = {}
        if "enabled" not in generation_cfg.colocated:
            generation_cfg.colocated.enabled = True
        if "resources" not in generation_cfg.colocated:
            generation_cfg.colocated.resources = {
                "gpus_per_node": None,
                "num_nodes": None,
            }
        if "vllm_cfg" not in generation_cfg:
            generation_cfg.vllm_cfg = {}
        vllm_cfg = generation_cfg.vllm_cfg
        if "pipeline_parallel_size" not in vllm_cfg:
            vllm_cfg.pipeline_parallel_size = 1
        if "expert_parallel_size" not in vllm_cfg:
            vllm_cfg.expert_parallel_size = 1
    return generation_cfg


def _configure_generation(generation_cfg, tokenizer):
    """v0.5.0 / Super3: configure_generation_config. Newer trees may ship a parser."""
    generation_cfg = _ensure_generation_defaults(generation_cfg)
    try:
        from nemo_rl.models.generation.vllm_config_parse import parse_vllm_tokenizer_mode
    except ImportError:
        from nemo_rl.models.generation import configure_generation_config

        return configure_generation_config(generation_cfg, tokenizer)
    parse_vllm_tokenizer_mode(generation_cfg, tokenizer)
    return generation_cfg


def _setup_nemo_gym_config(config, tokenizer) -> None:
    import inspect

    from nemo_rl.environments.nemo_gym import setup_nemo_gym_config

    params = inspect.signature(setup_nemo_gym_config).parameters
    if "tokenizer" in params or len(params) >= 2:
        setup_nemo_gym_config(config, tokenizer)
    else:
        setup_nemo_gym_config(config)


def _install_and_rewrite_gym_paths(config, project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from install_gym_server import find_gym_root, gym_config_paths, install_nemotron_verifier

    gym_root = find_gym_root()
    install_nemotron_verifier(gym_root=gym_root, project_root=project_root)
    os.environ.setdefault("NEMO_GYM_ROOT", str(gym_root))
    os.environ.setdefault("NEMOTRON_RLVR_ROOT", str(project_root))

    from omegaconf import open_dict

    with open_dict(config):
        if "env" not in config:
            config.env = {}
        if "nemo_gym" not in config.env:
            config.env.nemo_gym = {}
        config.env.nemo_gym.config_paths = gym_config_paths(gym_root, project_root)
    print(f"[train] Gym root: {gym_root}")
    print(f"[train] config_paths: {list(config.env.nemo_gym.config_paths)}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _maybe_register_omegaconf_resolvers()

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import setup, grpo_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.virtual_cluster import init_ray
    from train_utils import prepare_nemo_gym_dataset

    config = OmegaConf.load(str(config_path))
    config = _apply_schema_defaults(config)
    OmegaConf.resolve(config)

    tokenizer = get_tokenizer(config.policy.tokenizer)
    config.policy.generation = _configure_generation(config.policy.generation, tokenizer)

    use_gym = _should_use_gym(config)
    if use_gym:
        _install_and_rewrite_gym_paths(config, project_root)
        _setup_nemo_gym_config(config, tokenizer)

    init_ray()

    import ray

    train_path = _jsonl_path(config.data, "train")
    val_path = _jsonl_path(config.data, "validation")
    if not train_path:
        raise KeyError("data.train.data_path (or data.train_jsonl_fpath) is required")
    if not val_path:
        raise KeyError("data.validation.data_path (or data.validation_jsonl_fpath) is required")

    train_dataset = prepare_nemo_gym_dataset(config, tokenizer, train_path)
    val_dataset = prepare_nemo_gym_dataset(config, tokenizer, val_path)
    _configure_val_sizes(config, val_dataset)

    # NeMo RL expects a plain dict (MasterConfig TypedDict), which is what the
    # official runner passes (OmegaConf.to_container -> MasterConfig(**cfg)).
    # setup() keeps this object as master_config and the checkpointer runs
    # yaml.safe_dump(master_config) at every save_period; PyYAML's SafeDumper
    # cannot represent an OmegaConf DictConfig -> RepresenterError and an empty
    # config.yaml in tmp_step_N. Convert after all open_dict() mutations above.
    config = OmegaConf.to_container(config, resolve=True)

    packed = _unpack_setup(setup(config, tokenizer, train_dataset, val_dataset))
    policy = packed["policy"]
    policy_generation = packed["policy_generation"]
    cluster = packed["cluster"]
    master_config = packed["master_config"]
    nemo_gym = packed["nemo_gym"]

    task_to_env = None
    if use_gym:
        if nemo_gym is None:
            # v0.5.0: setup() does not create the Gym actor.
            from nemo_rl.environments.nemo_gym import NemoGym, NemoGymConfig

            ng_cfg = master_config["env"]["nemo_gym"]
            if not isinstance(ng_cfg, dict):
                ng_cfg = OmegaConf.to_container(ng_cfg, resolve=True)
            nemo_gym_config = NemoGymConfig(**ng_cfg)
            nemo_gym = NemoGym.options(
                runtime_env={"py_executable": sys.executable}
            ).remote(nemo_gym_config)
            packed["nemo_gym"] = nemo_gym
        if hasattr(nemo_gym, "health_check"):
            ray.get(nemo_gym.health_check.remote())
        task_to_env = {"nemo_gym": nemo_gym}

    try:
        _call_grpo_train(grpo_train, packed, tokenizer, task_to_env)
    finally:
        # Teardown order matters. Actors must be shut down gracefully *before*
        # their placement groups are removed; otherwise Ray kills the workers
        # (INTENDED_SYSTEM_EXIT "placement group was removed"), and the later
        # Policy.__del__ / VllmGeneration.__del__ safety nets hit ActorDiedError
        # while trying to run the cleanup RPC on already-dead actors.
        _shutdown_actor("nemo_gym", nemo_gym)
        _shutdown_actor("policy_generation", policy_generation)
        _shutdown_actor("policy", policy)
        _shutdown_cluster(cluster)


def _shutdown_actor(name: str, obj) -> None:
    if obj is None:
        return
    try:
        # Ray actor handle (NemoGym) vs. driver-side wrapper (Policy, VllmGeneration).
        if hasattr(obj, "shutdown") and hasattr(getattr(obj, "shutdown"), "remote"):
            import ray

            ray.get(obj.shutdown.remote(), timeout=60)
        elif callable(getattr(obj, "shutdown", None)):
            obj.shutdown()
    except Exception as exc:
        print(f"[train] {name} shutdown: {exc}")


if __name__ == "__main__":
    main()
