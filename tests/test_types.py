"""Regression tests for the typed records exchanged between pipeline stages."""

from __future__ import annotations

from drone3d.types import TelemetrySample


def test_telemetry_sample_to_dict_works_with_slots() -> None:
    """F1: ``@dataclass(slots=True)`` has no ``__dict__``; to_dict must still work."""
    sample = TelemetrySample(t=1.5, lat=12.5, lon=77.6)

    data = sample.to_dict()

    assert data == {"t": 1.5, "lat": 12.5, "lon": 77.6, "source": "unknown"}


def test_telemetry_sample_to_dict_omits_unset_optionals() -> None:
    sample = TelemetrySample(t=0.0, source="srt")

    data = sample.to_dict()

    assert data == {"t": 0.0, "source": "srt"}
    for key in ("lat", "lon", "alt_m", "rel_alt_m", "heading_deg", "speed_ms"):
        assert key not in data


def test_telemetry_sample_to_dict_includes_explicit_zero_values() -> None:
    """A falsy-but-set value must survive the None filter."""
    sample = TelemetrySample(t=0.0, lat=0.0, lon=0.0, alt_m=0.0, speed_ms=0.0)

    data = sample.to_dict()

    assert data["lat"] == 0.0
    assert data["lon"] == 0.0
    assert data["alt_m"] == 0.0
    assert data["speed_ms"] == 0.0


def test_telemetry_sample_has_position() -> None:
    assert TelemetrySample(t=0.0, lat=1.0, lon=2.0).has_position()
    assert not TelemetrySample(t=0.0, lat=1.0).has_position()
    assert not TelemetrySample(t=0.0).has_position()
