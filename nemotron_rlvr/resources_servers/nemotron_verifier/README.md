# `nemotron_verifier` Gym resources server

One FastAPI process that scores P0–P6 generations by calling the existing
Python verifiers in `nemotron_rlvr/verifiers/`.

Gym does **not** load this directory from the git clone. `setup_env_command`
opens:

```
$GYM_ROOT/resources_servers/nemotron_verifier/pyproject.toml
```

Ship **only** `pyproject.toml` (no `requirements.txt` — Gym errors if both exist).

## Install into Gym

From `nemotron_rlvr/`:

```bash
python3 scripts/install_gym_server.py
# or
NEMO_GYM_ROOT=/opt/nemo-rl/3rdparty/Gym-workspace/Gym python3 scripts/install_gym_server.py
```

`train.py` runs the same install automatically before Gym spinup.

## Config

`env.nemo_gym.config_paths` should be:

1. `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`
2. `resources_servers/nemotron_verifier/configs/nemotron_verifier.yaml`

JSONL rows must include `responses_create_params.input`, the extra verifier
fields (`expected_answer`, `task_name` / `family`, `verifier_meta`), and:

```json
"agent_ref": {"type": "responses_api_agents", "name": "nemotron_verifier_simple_agent"}
```

`scripts/reformat_dataset.py` writes that schema.
