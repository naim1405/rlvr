#!/usr/bin/env python3
"""Main RLVR training launcher for 7-family problem set (P0-P6)."""

import argparse
import os
import pprint
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run RLVR training for P0-P6 problems")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args, overrides = parser.parse_known_args()

    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)

    print(f"Executing RLVR training from root: {project_root}")
    print(f"Loading configuration file: {args.config}")

    import ray
    from nemo_rl.algorithms.grpo import (
        _should_use_nemo_gym,
        grpo_train,
        setup,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.nemo_gym import (
        NemoGym,
        NemoGymConfig,
        setup_nemo_gym_config,
    )
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from omegaconf import OmegaConf

    config = load_config(args.config)
    if overrides:
        print(f"Applying Hydra overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config_container = OmegaConf.to_container(config, resolve=True)
    
    tokenizer = get_tokenizer(config_container["policy"]["tokenizer"])
    config_container["policy"]["generation"] = configure_generation_config(
        config_container["policy"]["generation"], tokenizer
    )

    setup_nemo_gym_config(config_container, tokenizer)
    assert _should_use_nemo_gym(config_container)

    print("\n--- Final NeMo RLVR Configuration ---")
    pprint.pprint(config_container)

    init_ray()

    from train_utils import load_nemo_gym_dataset
    train_dataset = load_nemo_gym_dataset(config_container["data"]["train"]["data_path"], tokenizer)
    val_dataset = load_nemo_gym_dataset(config_container["data"]["validation"]["data_path"], tokenizer)

    config_container["grpo"]["max_val_samples"] = len(val_dataset)
    config_container["grpo"]["val_batch_size"] = len(val_dataset)

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
    ) = setup(config_container, tokenizer, train_dataset, val_dataset)

    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config_container["env"]["nemo_gym"],
    )
    nemo_gym = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym"),
        }
    ).remote(nemo_gym_config)
    
    ray.get(nemo_gym.health_check.remote())
    task_to_env = {"nemo_gym": nemo_gym}

    print("\n>>> Starting Async-GRPO Training Loop...")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

if __name__ == "__main__":
    main()
