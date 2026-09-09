"""GRPO training entrypoint for Qwen + NeMo Gym (P0–P6 `nemotron_verifier`)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")


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


def _looks_like_gym_actor(obj) -> bool:
    if obj is None:
        return False
    return any(hasattr(obj, name) for name in ("health_check", "prepare_for_generation", "get_reward"))


def _unpack_setup(result):
    """v0.5.0 setup() is a 10-tuple. Super3/main insert nemo_gym as the 3rd value."""
    result = tuple(result)
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
    if len(result) >= 11 and _looks_like_gym_actor(result[2]):
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
        "expected 10 (v0.5.0) or ≥11 with a Gym actor at index 2."
    )


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

    packed = _unpack_setup(setup(config, tokenizer, train_dataset, val_dataset))
    policy = packed["policy"]
    policy_generation = packed["policy_generation"]
    cluster = packed["cluster"]
    dataloader = packed["dataloader"]
    val_dataloader = packed["val_dataloader"]
    logger = packed["logger"]
    checkpointer = packed["checkpointer"]
    grpo_state = packed["grpo_state"]
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
        ray.get(nemo_gym.health_check.remote())
        task_to_env = {"nemo_gym": nemo_gym}

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        task_to_env,
    )

    if hasattr(policy, "shutdown_collective_rpc"):
        policy.shutdown_collective_rpc()
    cluster.shutdown()


if __name__ == "__main__":
    main()
