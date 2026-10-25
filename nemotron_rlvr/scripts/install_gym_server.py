#!/usr/bin/env python3
"""Copy the nemotron_verifier resources server into a NeMo Gym checkout.

Gym's setup_env_command looks up:

    $GYM_ROOT/resources_servers/<yaml-inner-key>/pyproject.toml

A copy that only lives in this repo is not enough. This script (also called
from train.py) copies:

    resources_servers/nemotron_verifier/{app.py,score.py,pyproject.toml,configs/}
    verifiers/

into $GYM_ROOT/resources_servers/nemotron_verifier/. Existing .venv is kept so
`skip_venv_if_present: true` still works after the first successful spinup.

Discovery order for $GYM_ROOT:
  1. NEMO_GYM_ROOT / GYM_ROOT / NEMO_GYM_DIR
  2. nemo_gym.PARENT_DIR (or the package's parent)
  3. /opt/nemo-rl/3rdparty/Gym-workspace/Gym  (Super3 container)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


SERVER_NAME = "nemotron_verifier"
_COPY_FILES = ("app.py", "score.py", "pyproject.toml", "README.md")
_DEFAULT_CANDIDATES = (
    Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym"),
    Path("/opt/nemo-rl/3rdparty/Gym"),
    Path("/opt/Gym"),
)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def _looks_like_gym_root(path: Path) -> bool:
    return (path / "resources_servers").is_dir()


def find_gym_root() -> Path:
    for key in ("NEMO_GYM_ROOT", "GYM_ROOT", "NEMO_GYM_DIR"):
        raw = os.environ.get(key)
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if _looks_like_gym_root(path):
            return path
        nested = path / "Gym"
        if _looks_like_gym_root(nested):
            return nested

    try:
        import nemo_gym

        parent_dir = getattr(nemo_gym, "PARENT_DIR", None)
        if parent_dir:
            path = Path(parent_dir)
            if _looks_like_gym_root(path):
                return path
        pkg = Path(nemo_gym.__file__).resolve().parent
        for candidate in (pkg.parent, pkg):
            if _looks_like_gym_root(candidate):
                return candidate
    except ImportError:
        pass

    for candidate in _DEFAULT_CANDIDATES:
        if _looks_like_gym_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot locate the NeMo Gym checkout (need a resources_servers/ directory). "
        "Set NEMO_GYM_ROOT to the Gym repo root, e.g. "
        "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
    )


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def install_nemotron_verifier(
    gym_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    project_root = (project_root or project_root_from_here()).resolve()
    gym_root = (gym_root or find_gym_root()).resolve()
    src = project_root / "resources_servers" / SERVER_NAME
    if not (src / "app.py").is_file() or not (src / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Gym server sources missing under {src} (need app.py and pyproject.toml)"
        )
    if (src / "requirements.txt").is_file():
        raise RuntimeError(
            f"{src / 'requirements.txt'} must not exist: Gym's setup_env_command "
            "rejects having both pyproject.toml and requirements.txt."
        )

    dest = gym_root / "resources_servers" / SERVER_NAME
    if dest.is_symlink():
        dest.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    for name in _COPY_FILES:
        src_file = src / name
        if src_file.is_file():
            _copy_file(src_file, dest / name)

    cfg_src = src / "configs"
    if cfg_src.is_dir():
        cfg_dest = dest / "configs"
        cfg_dest.mkdir(exist_ok=True)
        for yaml_file in cfg_src.glob("*.yaml"):
            _copy_file(yaml_file, cfg_dest / yaml_file.name)

    verifiers_src = project_root / "verifiers"
    if not (verifiers_src / "p0_retrieval.py").is_file():
        raise FileNotFoundError(f"Missing verifiers at {verifiers_src}")
    verifiers_dest = dest / "verifiers"
    if verifiers_dest.is_symlink() or verifiers_dest.is_file():
        verifiers_dest.unlink()
    shutil.copytree(
        verifiers_src,
        verifiers_dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    pyproject = dest / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"Install failed: {pyproject} was not written")
    print(f"[install_gym_server] wrote {dest}")
    print(f"[install_gym_server] pyproject.toml -> {pyproject}")
    return dest


def gym_config_paths(gym_root: Path, project_root: Path) -> list[str]:
    """Absolute yaml paths Gym should load for this environment."""
    bundled = (
        gym_root
        / "responses_api_models"
        / "vllm_model"
        / "configs"
        / "vllm_model_for_training.yaml"
    )
    local_policy = project_root / "configs" / "models" / "policy_model.yaml"
    policy = bundled if bundled.is_file() else local_policy

    installed = (
        gym_root / "resources_servers" / SERVER_NAME / "configs" / f"{SERVER_NAME}.yaml"
    )
    local_server = (
        project_root / "resources_servers" / SERVER_NAME / "configs" / f"{SERVER_NAME}.yaml"
    )
    server = installed if installed.is_file() else local_server
    return [str(policy), str(server)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gym-root",
        type=Path,
        default=None,
        help="NeMo Gym checkout (default: autodetect / $NEMO_GYM_ROOT)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="nemotron_rlvr/ directory (default: parent of this script)",
    )
    args = parser.parse_args(argv)
    dest = install_nemotron_verifier(gym_root=args.gym_root, project_root=args.project_root)
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
