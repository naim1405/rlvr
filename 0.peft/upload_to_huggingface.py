#!/usr/bin/env python3
"""Upload a local folder (usually a PEFT adapter ``model/`` dir) to Hugging Face.

This is intentionally separate from training: ``run_peft_training.sh`` /
``run_in_container.sh`` never upload anything. Run this only after a training
job has finished and you have chosen the adapter directory to publish or
archive, e.g. ``0.peft/checkpoints/epoch_1/model``.

Token handling (first match wins):

1. ``--token`` (accepted for convenience, but it can land in shell history /
   process lists -- prefer one of the options below),
2. ``--token-file PATH`` (a file outside the repo containing the token),
3. the environment variable named by ``--token-env`` (default ``HF_TOKEN``,
   with ``HUGGING_FACE_HUB_TOKEN`` as a legacy fallback),
4. an interactive hidden prompt, if stdin is a TTY.

Examples::

    # Recommended: token in the environment, repo given as a URL.
    export HF_TOKEN='hf_...'   # a WRITE token; never commit it
    python3 0.peft/upload_to_huggingface.py \
        --repo-url https://huggingface.co/<you>/<private-adapter-repo> \
        --upload-folder 0.peft/checkpoints/epoch_1/model \
        --require-adapter

    # Same thing with a repo id, creating the (private) repo if needed:
    python3 0.peft/upload_to_huggingface.py \
        --repo-id <you>/<private-adapter-repo> --create-repo \
        --upload-folder 0.peft/checkpoints/epoch_1/model --require-adapter

    # Validate everything without touching the network:
    python3 0.peft/upload_to_huggingface.py \
        --repo-id <you>/<private-adapter-repo> \
        --upload-folder 0.peft/checkpoints/epoch_1/model \
        --require-adapter --dry-run
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

REPO_TYPES = ("model", "dataset", "space")
_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)?$"
)
_URL_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
_TYPE_PREFIXES = {"models": "model", "datasets": "dataset", "spaces": "space"}
_TOKEN_ENV_NAMES = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}


class UserError(Exception):
    """An error message safe to show directly to the user."""


@dataclass(frozen=True)
class RepoRef:
    repo_id: str
    repo_type: str = "model"


def _validate_repo_id(repo_id: str) -> str:
    if not repo_id or not _REPO_ID_RE.match(repo_id) or ".." in repo_id:
        raise UserError(
            f"Invalid Hugging Face repo id: {repo_id!r}. "
            "Expected '<namespace>/<repo-name>', e.g. 'alice/qwen3-peft-adapter'."
        )
    return repo_id


def parse_repo_ref(value: str, repo_type: str | None = None) -> RepoRef:
    """Parse a repo URL or repo id into a :class:`RepoRef`.

    Accepts ``https://huggingface.co/<ns>/<repo>`` (optionally with a
    ``datasets/``/``spaces/`` prefix or a trailing ``/tree/<rev>`` etc.),
    ``hf.co/<ns>/<repo>``, and plain ``<ns>/<repo>`` ids. An explicitly
    passed ``repo_type`` overrides anything inferred from the value.
    """
    text = (value or "").strip()
    if not text:
        raise UserError("A Hugging Face repo URL or repo id is required.")

    explicit_type = repo_type.strip().lower() if repo_type else None
    if explicit_type and explicit_type not in REPO_TYPES:
        raise UserError(f"Invalid --repo-type {explicit_type!r}; choose one of {REPO_TYPES}.")

    inferred_type: str | None = None
    candidate = text

    looks_like_url = "://" in text or text.split("/", 1)[0].lower() in _URL_HOSTS
    if looks_like_url:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        segments = [unquote(part) for part in parsed.path.split("/") if part]
        if segments and segments[0].lower() in _TYPE_PREFIXES:
            inferred_type = _TYPE_PREFIXES[segments[0].lower()]
            segments = segments[1:]
        if len(segments) < 2:
            raise UserError(
                f"Could not find '<namespace>/<repo>' in repo URL: {value!r}."
            )
        # Tolerate pasted /tree/<rev>, /blob/<rev>/... style URLs.
        candidate = f"{segments[0]}/{segments[1]}"
    else:
        candidate = text.strip("/")
        parts = candidate.split("/")
        if len(parts) >= 3 and parts[0].lower() in _TYPE_PREFIXES:
            inferred_type = _TYPE_PREFIXES[parts[0].lower()]
            candidate = "/".join(parts[1:3])

    return RepoRef(
        repo_id=_validate_repo_id(candidate),
        repo_type=explicit_type or inferred_type or "model",
    )


def resolve_token(args: argparse.Namespace, environ: dict | None = None, *, allow_prompt: bool = False) -> str:
    """Resolve the HF token from CLI args / file / environment / prompt."""
    env = os.environ if environ is None else environ

    token = (getattr(args, "token", None) or "").strip()
    if token:
        print(
            "WARNING: --token puts the token in your shell history and process "
            "list. Prefer HF_TOKEN or --token-file.",
            file=sys.stderr,
        )
        return token

    token_file = getattr(args, "token_file", None)
    if token_file:
        path = Path(token_file).expanduser()
        if not path.is_file():
            raise UserError(f"--token-file does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line and line.split("=", 1)[0].strip() in _TOKEN_ENV_NAMES:
                line = line.split("=", 1)[1].strip().strip("'\"")
            if line:
                return line
        raise UserError(f"--token-file is empty: {path}")

    token_env = getattr(args, "token_env", None) or "HF_TOKEN"
    env_names = [token_env]
    if token_env == "HF_TOKEN":
        env_names.append("HUGGING_FACE_HUB_TOKEN")
    for name in env_names:
        token = (env.get(name) or "").strip()
        if token:
            return token

    if allow_prompt and sys.stdin.isatty():
        token = getpass.getpass("Hugging Face write token (input hidden): ").strip()
        if token:
            return token

    raise UserError(
        "No Hugging Face token found. Set HF_TOKEN to a WRITE token "
        "(recommended), pass --token-file <path outside the repo>, or run "
        "interactively to be prompted. Never commit the token."
    )


def normalize_path_in_repo(value: str | None) -> str | None:
    """Normalise --path-in-repo; the repo root becomes ``None`` (HfApi default)."""
    if value is None:
        return None
    text = value.strip().replace("\\", "/")
    if text in ("", ".", "/"):
        return None
    path = PurePosixPath(text.lstrip("/"))
    if any(part == ".." for part in path.parts):
        raise UserError(f"--path-in-repo must stay inside the repo: {value!r}")
    normalised = str(path)
    return None if normalised == "." else normalised


def folder_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file())


def human_bytes(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover


def validate_upload_folder(folder: Path, *, require_adapter: bool = False) -> tuple[list[Path], list[str]]:
    """Validate the upload folder; returns (files, warnings)."""
    if not folder.exists():
        raise UserError(f"Upload folder does not exist: {folder}")
    if not folder.is_dir():
        raise UserError(f"Upload folder is not a directory: {folder}")
    files = folder_files(folder)
    if not files:
        raise UserError(f"Upload folder contains no files: {folder}")

    warnings: list[str] = []
    if (folder / "optim").exists() or (folder / "model").is_dir() and (folder / "rng").exists():
        warnings.append(
            "this looks like a full epoch_* checkpoint (optim/rng state). For a "
            "PEFT adapter upload, point --upload-folder at its 'model' "
            "subdirectory instead."
        )

    has_adapter_config = (folder / "adapter_config.json").is_file()
    has_safetensors = any(p.suffix == ".safetensors" for p in files)
    if require_adapter:
        if not has_adapter_config:
            raise UserError(
                f"--require-adapter: no adapter_config.json at {folder}. "
                "Point --upload-folder at the checkpoint's model/ directory."
            )
        if not has_safetensors:
            raise UserError(
                f"--require-adapter: no .safetensors weights under {folder}."
            )
    elif not has_adapter_config:
        warnings.append(
            "no adapter_config.json at the upload root; if this is meant to be "
            "a PEFT adapter, upload epoch_*/model/ (and consider --require-adapter)."
        )
    return files, warnings


def repo_display_url(repo: RepoRef, revision: str | None = None, path_in_repo: str | None = None) -> str:
    prefix = {"dataset": "datasets/", "space": "spaces/"}.get(repo.repo_type, "")
    url = f"https://huggingface.co/{prefix}{repo.repo_id}"
    if path_in_repo:
        url += f"/tree/{revision or 'main'}/{path_in_repo}"
    elif revision:
        url += f"/tree/{revision}"
    return url


def build_upload_kwargs(args: argparse.Namespace, repo: RepoRef, folder: Path, token: str) -> dict:
    kwargs: dict = {
        "folder_path": str(folder),
        "repo_id": repo.repo_id,
        "repo_type": repo.repo_type,
        "token": token,
        "commit_message": args.commit_message,
    }
    optional = {
        "path_in_repo": normalize_path_in_repo(getattr(args, "path_in_repo", None)),
        "revision": getattr(args, "revision", None),
        "commit_description": getattr(args, "commit_description", None),
        "allow_patterns": getattr(args, "allow_pattern", None) or None,
        "ignore_patterns": getattr(args, "ignore_pattern", None) or None,
    }
    kwargs.update({k: v for k, v in optional.items() if v is not None})
    return kwargs


def upload_folder_to_hub(args: argparse.Namespace, repo: RepoRef, folder: Path, token: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise UserError(
            "The 'huggingface_hub' package is not installed here. Install it "
            "with 'python3 -m pip install huggingface_hub', or run this "
            "script where it is available (it is inside the NeMo AutoModel "
            "container)."
        ) from exc

    endpoint = getattr(args, "endpoint", None)
    api = HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)

    if getattr(args, "create_repo", False):
        create_kwargs: dict = {
            "repo_id": repo.repo_id,
            "repo_type": repo.repo_type,
            "private": bool(getattr(args, "private", True)),
            "exist_ok": True,
            "token": token,
        }
        space_sdk = getattr(args, "space_sdk", None)
        if space_sdk:
            create_kwargs["space_sdk"] = space_sdk
        if repo.repo_type == "space" and not space_sdk:
            raise UserError(
                "Creating a Space requires --space-sdk (e.g. gradio); create "
                "the Space first and drop --create-repo."
            )
        print(
            f"Ensuring repo exists: {repo.repo_id} "
            f"({'private' if create_kwargs['private'] else 'public'}, type={repo.repo_type})"
        )
        api.create_repo(**create_kwargs)

    result = api.upload_folder(**build_upload_kwargs(args, repo, folder, token))
    return str(result)


def _redact(message: str, token: str | None) -> str:
    return message.replace(token, "***") if token else message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a local folder (e.g. a PEFT adapter model/ directory) "
        "to a Hugging Face repo. Separate from training; nothing uploads "
        "unless you run this script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo_group = parser.add_argument_group("destination repo (exactly one of --repo-url / --repo-id)")
    repo_group.add_argument(
        "--repo-url",
        default=None,
        help="Repo URL, e.g. https://huggingface.co/<you>/<repo> (or set HF_REPO_URL).",
    )
    repo_group.add_argument(
        "--repo-id",
        default=None,
        help="Repo id, e.g. <you>/<repo> (or set HF_REPO_ID).",
    )
    parser.add_argument(
        "--repo-type",
        choices=REPO_TYPES,
        default=None,
        help="Repo type. Inferred from a datasets//spaces/ URL prefix; default: model.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Folder inside the repo to upload into. Default/'.' = repo root.",
    )
    parser.add_argument("--revision", default=None, help="Branch/revision to commit to.")
    parser.add_argument("--commit-message", default="Upload PEFT adapter", help="Commit message.")
    parser.add_argument("--commit-description", default=None, help="Optional longer commit description.")
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo first if it does not exist (exist_ok). Off by default: prefer pre-creating it.",
    )
    parser.add_argument(
        "--private",
        dest="private",
        action="store_true",
        default=True,
        help="With --create-repo: create the repo private (default). Does not change an existing repo.",
    )
    parser.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="With --create-repo: create the repo public. Use deliberately.",
    )
    parser.add_argument("--space-sdk", default=None, help="Only for --create-repo with --repo-type space.")

    parser.add_argument(
        "--upload-folder",
        default=os.environ.get("UPLOAD_FOLDER"),
        help="Local folder to upload, e.g. 0.peft/checkpoints/epoch_1/model (or set UPLOAD_FOLDER).",
    )
    parser.add_argument(
        "--require-adapter",
        action="store_true",
        help="Fail unless the folder root has adapter_config.json and .safetensors weights.",
    )
    parser.add_argument("--allow-pattern", action="append", default=None, help="Glob of files to include (repeatable).")
    parser.add_argument("--ignore-pattern", action="append", default=None, help="Glob of files to exclude (repeatable).")

    token_group = parser.add_argument_group("authentication (never commit the token)")
    token_group.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable holding the HF write token.",
    )
    token_group.add_argument("--token-file", default=None, help="File (outside the repo) containing the HF write token.")
    token_group.add_argument(
        "--token",
        default=None,
        help="HF write token value. Discouraged: visible in shell history/process list.",
    )
    parser.add_argument("--endpoint", default=None, help="Custom HF endpoint (default: huggingface.co / HF_ENDPOINT).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded and exit; no token or network needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    token: str | None = None
    try:
        # Explicit CLI flags win; only fall back to the environment when
        # neither was given, so a stray HF_REPO_ID can't clash with an
        # explicit --repo-url (or vice versa).
        if not args.repo_url and not args.repo_id:
            args.repo_url = os.environ.get("HF_REPO_URL") or None
            args.repo_id = os.environ.get("HF_REPO_ID") or None
        if bool(args.repo_url) == bool(args.repo_id):
            parser.error("exactly one destination is required: --repo-url or --repo-id (or set HF_REPO_URL / HF_REPO_ID)")
        repo = parse_repo_ref(args.repo_url or args.repo_id, args.repo_type)

        if not args.upload_folder:
            parser.error("--upload-folder is required (or set UPLOAD_FOLDER)")
        folder = Path(args.upload_folder).expanduser()
        files, warnings = validate_upload_folder(folder, require_adapter=args.require_adapter)
        total_bytes = sum(p.stat().st_size for p in files)
        path_in_repo = normalize_path_in_repo(args.path_in_repo)

        print(f"Upload folder : {folder} ({len(files)} files, {human_bytes(total_bytes)})")
        print(
            f"Destination   : {repo_display_url(repo, args.revision, path_in_repo)} "
            f"(repo_id={repo.repo_id}, type={repo.repo_type})"
        )
        print(f"Create repo   : {args.create_repo}" + (f" (visibility: {'private' if args.private else 'public'})" if args.create_repo else ""))
        if args.allow_pattern:
            print(f"Allow patterns: {args.allow_pattern}")
        if args.ignore_pattern:
            print(f"Ignore patterns: {args.ignore_pattern}")
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        if args.dry_run:
            print("DRY RUN: no token used, nothing uploaded.")
            for path in files:
                print(f"  would upload: {path.relative_to(folder).as_posix()}")
            return 0

        token = resolve_token(args, allow_prompt=True)
        result_url = upload_folder_to_hub(args, repo, folder, token)
        print(f"Upload complete: {result_url}")
        return 0
    except UserError as exc:
        print(f"ERROR: {_redact(str(exc), token)}", file=sys.stderr)
        return 2
    except Exception as exc:  # HF/network errors: report, never leak the token.
        print(f"ERROR: upload failed: {_redact(str(exc), token)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
