#!/usr/bin/env python3
"""CPU-only tests for upload_to_huggingface.py.

No network and no real token are needed: the HfApi path is exercised with a
fake ``huggingface_hub`` module injected into ``sys.modules``.

Run from the repo root:
    python3 0.peft/test_upload_to_huggingface.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import upload_to_huggingface as up  # noqa: E402


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        token=None,
        token_file=None,
        token_env="HF_TOKEN",
        repo_url=None,
        repo_id=None,
        repo_type=None,
        upload_folder=None,
        path_in_repo=None,
        revision=None,
        commit_message="Upload PEFT adapter",
        commit_description=None,
        create_repo=False,
        private=True,
        space_sdk=None,
        require_adapter=False,
        allow_pattern=None,
        ignore_pattern=None,
        endpoint=None,
        dry_run=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_adapter_dir(root: Path) -> Path:
    adapter = root / "epoch_0_step_31" / "model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    return adapter


class ParseRepoRefTests(unittest.TestCase):
    def test_plain_repo_id(self):
        self.assertEqual(up.parse_repo_ref("alice/qwen3-peft"), up.RepoRef("alice/qwen3-peft", "model"))

    def test_model_url(self):
        ref = up.parse_repo_ref("https://huggingface.co/alice/qwen3-peft")
        self.assertEqual(ref.repo_id, "alice/qwen3-peft")
        self.assertEqual(ref.repo_type, "model")

    def test_url_with_tree_suffix_and_dataset_prefix(self):
        ref = up.parse_repo_ref("https://huggingface.co/datasets/alice/data/tree/main/sub")
        self.assertEqual(ref, up.RepoRef("alice/data", "dataset"))

    def test_hf_co_short_host(self):
        self.assertEqual(up.parse_repo_ref("hf.co/alice/qwen3-peft").repo_id, "alice/qwen3-peft")

    def test_explicit_type_overrides_inference(self):
        ref = up.parse_repo_ref("https://huggingface.co/datasets/alice/data", repo_type="model")
        self.assertEqual(ref.repo_type, "model")

    def test_invalid_values_raise(self):
        for bad in ("", "https://huggingface.co/alice", "alice//repo", "alice/../repo"):
            with self.subTest(bad=bad), self.assertRaises(up.UserError):
                up.parse_repo_ref(bad)


class TokenResolutionTests(unittest.TestCase):
    def test_env_token(self):
        self.assertEqual(up.resolve_token(make_args(), {"HF_TOKEN": "hf_test"}), "hf_test")

    def test_legacy_env_fallback(self):
        self.assertEqual(
            up.resolve_token(make_args(), {"HUGGING_FACE_HUB_TOKEN": "hf_old"}),
            "hf_old",
        )

    def test_custom_token_env(self):
        args = make_args(token_env="MY_HF_WRITE_TOKEN")
        self.assertEqual(up.resolve_token(args, {"MY_HF_WRITE_TOKEN": "hf_x"}), "hf_x")
        with self.assertRaises(up.UserError):
            up.resolve_token(args, {"HF_TOKEN": "hf_wrong_var"})

    def test_token_file_and_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "token"
            token_file.write_text("# comment\nHF_TOKEN=hf_from_file\n", encoding="utf-8")
            args = make_args(token_file=str(token_file))
            self.assertEqual(up.resolve_token(args, {"HF_TOKEN": "hf_env"}), "hf_from_file")

    def test_cli_token_wins_but_warns(self):
        args = make_args(token="hf_cli")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(up.resolve_token(args, {}), "hf_cli")
        self.assertIn("WARNING", err.getvalue())
        self.assertNotIn("hf_cli", err.getvalue())

    def test_missing_token_raises_without_prompt(self):
        with self.assertRaises(up.UserError):
            up.resolve_token(make_args(), {})


class FolderValidationTests(unittest.TestCase):
    def test_require_adapter_accepts_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            files, warnings = up.validate_upload_folder(adapter, require_adapter=True)
            self.assertEqual(len(files), 2)
            self.assertEqual(warnings, [])

    def test_require_adapter_rejects_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            epoch_dir = adapter.parent
            (epoch_dir / "optim").mkdir()
            with self.assertRaises(up.UserError):
                up.validate_upload_folder(epoch_dir, require_adapter=True)
            _, warnings = up.validate_upload_folder(epoch_dir, require_adapter=False)
            self.assertTrue(any("full epoch_* checkpoint" in w for w in warnings))

    def test_missing_and_empty_folders_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(up.UserError):
                up.validate_upload_folder(Path(tmp) / "nope")
            with self.assertRaises(up.UserError):
                up.validate_upload_folder(Path(tmp))


class PathInRepoTests(unittest.TestCase):
    def test_root_becomes_none(self):
        for value in (None, "", ".", "/"):
            self.assertIsNone(up.normalize_path_in_repo(value))

    def test_subfolder_kept(self):
        self.assertEqual(up.normalize_path_in_repo("/adapters/step-31"), "adapters/step-31")

    def test_escape_rejected(self):
        with self.assertRaises(up.UserError):
            up.normalize_path_in_repo("../outside")


class FakeHfApi:
    created: list[dict] = []
    uploaded: list[dict] = []
    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def create_repo(self, **kwargs):
        type(self).created.append(kwargs)
        return "created"

    def upload_folder(self, **kwargs):
        type(self).uploaded.append(kwargs)
        return f"https://huggingface.co/{kwargs['repo_id']}"


class UploadCallTests(unittest.TestCase):
    def setUp(self):
        FakeHfApi.created = []
        FakeHfApi.uploaded = []
        FakeHfApi.init_kwargs = {}
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.HfApi = FakeHfApi
        self._patcher = mock.patch.dict(sys.modules, {"huggingface_hub": fake_module})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_upload_without_create_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            args = make_args()
            repo = up.RepoRef("alice/qwen3-peft", "model")
            with contextlib.redirect_stdout(io.StringIO()):
                url = up.upload_folder_to_hub(args, repo, adapter, "hf_secret")
            self.assertEqual(url, "https://huggingface.co/alice/qwen3-peft")
            self.assertEqual(FakeHfApi.created, [])
            self.assertEqual(len(FakeHfApi.uploaded), 1)
            call = FakeHfApi.uploaded[0]
            self.assertEqual(call["repo_id"], "alice/qwen3-peft")
            self.assertEqual(call["folder_path"], str(adapter))
            self.assertEqual(call["token"], "hf_secret")
            self.assertNotIn("path_in_repo", call)  # default = repo root

    def test_create_repo_is_private_by_default_and_path_in_repo_is_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            args = make_args(create_repo=True, path_in_repo="adapters/final", revision="main")
            repo = up.RepoRef("alice/qwen3-peft", "model")
            with contextlib.redirect_stdout(io.StringIO()):
                up.upload_folder_to_hub(args, repo, adapter, "hf_secret")
            self.assertEqual(len(FakeHfApi.created), 1)
            self.assertTrue(FakeHfApi.created[0]["private"])
            self.assertTrue(FakeHfApi.created[0]["exist_ok"])
            self.assertEqual(FakeHfApi.uploaded[0]["path_in_repo"], "adapters/final")
            self.assertEqual(FakeHfApi.uploaded[0]["revision"], "main")


class MainDryRunTests(unittest.TestCase):
    def test_dry_run_needs_no_token_and_uploads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            argv = [
                "--repo-url", "https://huggingface.co/alice/qwen3-peft",
                "--upload-folder", str(adapter),
                "--require-adapter",
                "--dry-run",
            ]
            with mock.patch.dict("os.environ", {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(up.main(argv), 0)
            text = out.getvalue()
            self.assertIn("DRY RUN", text)
            self.assertIn("adapter_config.json", text)

    def test_explicit_repo_url_beats_env_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_dir(Path(tmp))
            argv = [
                "--repo-url", "https://huggingface.co/alice/from-cli",
                "--upload-folder", str(adapter),
                "--dry-run",
            ]
            env = {"HF_REPO_ID": "bob/from-env"}
            with mock.patch.dict("os.environ", env, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(up.main(argv), 0)
            self.assertIn("alice/from-cli", out.getvalue())
            self.assertNotIn("bob/from-env", out.getvalue())


if __name__ == "__main__":
    unittest.main()
