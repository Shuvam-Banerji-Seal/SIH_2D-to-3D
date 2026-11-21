"""Where model weights are looked up: the Marigold assets and the Hugging Face / torch caches."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from drone3d.depth import assets
from drone3d.utils import model_cache


def test_assets_dir_prefers_an_explicit_path_then_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DEPTH_ASSETS_DIR", raising=False)
    assert assets.resolve_assets_dir() == assets.DEFAULT_ASSETS

    monkeypatch.setenv("DEPTH_ASSETS_DIR", str(tmp_path / "env"))
    assert assets.resolve_assets_dir() == tmp_path / "env"
    assert assets.resolve_assets_dir(tmp_path / "arg") == tmp_path / "arg"
    assert assets.resolve_assets_dir(str(tmp_path / "arg")) == tmp_path / "arg"


def test_assets_dir_needs_no_torch() -> None:
    code = "import sys; sys.modules['torch'] = None; import drone3d.depth.assets"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_weight_store_is_used_only_when_the_machine_has_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for var in ("HF_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(model_cache, "STORE", tmp_path / "absent")
    model_cache.use_weight_store()
    assert "HF_HUB_CACHE" not in os.environ and "TORCH_HOME" not in os.environ

    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(model_cache, "STORE", store)
    model_cache.use_weight_store()
    assert os.environ["HF_HUB_CACHE"] == str(store)
    assert os.environ["TORCH_HOME"] == str(store / "torch")


def test_weight_store_never_overrides_what_the_user_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(model_cache, "STORE", tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", "/elsewhere")
    monkeypatch.setenv("TORCH_HOME", "/elsewhere/torch")
    model_cache.use_weight_store()
    assert os.environ["HF_HUB_CACHE"] == "/elsewhere"
    assert os.environ["TORCH_HOME"] == "/elsewhere/torch"


def test_hf_hub_cache_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(model_cache, "STORE", tmp_path / "absent")
    assert model_cache.hf_hub_cache() == tmp_path / "home" / "hub"

    monkeypatch.setattr(model_cache, "STORE", tmp_path)
    assert model_cache.hf_hub_cache() == tmp_path

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert model_cache.hf_hub_cache() == tmp_path / "explicit"
