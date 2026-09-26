"""Tests for configuration loading, ``--set`` overrides and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.config import (
    PipelineConfig,
    apply_override,
    load_config,
)
from drone3d.exceptions import ConfigError

# --- loading ---------------------------------------------------------------


def test_defaults_when_no_path() -> None:
    config = load_config(None)

    assert isinstance(config, PipelineConfig)
    assert config.sfm.backend == "colmap"
    assert config.preprocess.max_frames > 0


def test_load_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("sfm:\n  max_features: 1234\n")

    config = load_config(path)

    assert config.sfm.max_features == 1234
    # untouched keys keep their defaults
    assert config.sfm.backend == "colmap"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a\n- list\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_empty_yaml_loads_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("")

    assert load_config(path).sfm.backend == "colmap"


def test_shipped_profiles_load() -> None:
    for name in ("default", "fast", "accurate"):
        config = load_config(Path("configs") / f"{name}.yaml")
        assert isinstance(config, PipelineConfig)


# --- overrides -------------------------------------------------------------


def test_override_sets_typed_value() -> None:
    config = load_config(None, ["preprocess.max_frames=25", "sfm.max_features=7"])

    assert config.preprocess.max_frames == 25
    assert config.sfm.max_features == 7


def test_override_float_and_bool() -> None:
    config = load_config(None, ["preprocess.min_spacing_s=1.5", "preprocess.dynamic_masking=true"])

    assert config.preprocess.min_spacing_s == 1.5
    assert config.preprocess.dynamic_masking is True


def test_override_bool_accepts_yes_no() -> None:
    config = load_config(None, ["preprocess.deblur=yes", "preprocess.stabilize=no"])

    assert config.preprocess.deblur is True
    assert config.preprocess.stabilize is False


def test_override_requires_equals_sign() -> None:
    with pytest.raises(ConfigError, match="key=value"):
        load_config(None, ["sfm.max_features"])


def test_override_unknown_key_raises() -> None:
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(None, ["sfm.nope=1"])


def test_override_unknown_section_raises() -> None:
    with pytest.raises(ConfigError, match="unknown configuration section"):
        load_config(None, ["nope.max_frames=1"])


def test_override_bad_int_raises() -> None:
    with pytest.raises(ConfigError, match="integer"):
        load_config(None, ["preprocess.max_frames=abc"])


def test_override_bad_bool_raises() -> None:
    with pytest.raises(ConfigError, match="boolean"):
        load_config(None, ["preprocess.deblur=maybe"])


def test_override_null_string_becomes_none() -> None:
    config = load_config(None, ["ingest.video="])

    assert config.ingest.video is None


def test_override_list_field_wraps_scalar() -> None:
    config = load_config(None, ["sfm.extra_args=--SiftExtraction.use_gpu"])

    assert config.sfm.extra_args == ["--SiftExtraction.use_gpu"]


def test_apply_override_on_unknown_key_raises() -> None:
    config = load_config(None)

    with pytest.raises(ConfigError):
        apply_override(config, "sfm.not_a_field", "1")


def test_apply_override_nested_section() -> None:
    config = load_config(None)

    apply_override(config, "geo.min_correspondences", "5")

    assert config.geo.min_correspondences == 5


# --- validation ------------------------------------------------------------


def test_validate_rejects_empty_stage_list(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("pipeline:\n  stages: []\n")

    with pytest.raises(ConfigError):
        load_config(path)


def test_validate_rejects_unknown_stage(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("pipeline:\n  stages: [ingest, bogus]\n")

    with pytest.raises(ConfigError):
        load_config(path)
