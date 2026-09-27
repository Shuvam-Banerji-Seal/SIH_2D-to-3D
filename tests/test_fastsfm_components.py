"""One pass can map as several components: every one with enough images is kept, and the fallback counts them all."""

from __future__ import annotations

from pathlib import Path

from drone3d.fastsfm import stage


class _Rec:
    def __init__(self, n: int) -> None:
        self.n = n

    def num_reg_images(self) -> int:
        return self.n


def test_components_keeps_every_large_enough_one_largest_first() -> None:
    comps = stage._components({0: _Rec(25), 1: _Rec(2), 2: _Rec(43), 3: _Rec(6)}, 3)
    assert [c.n for c in comps] == [43, 25, 6]


def test_no_fallback_when_components_together_cover_the_pass(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []

    def fake_map(db, images, out, *, mapper, verify=True):  # type: ignore[no-untyped-def]
        calls.append(mapper)
        return {0: _Rec(43), 1: _Rec(25), 2: _Rec(6)}, {"verification_s": 0.1, "mapping_s": 1.0}

    monkeypatch.setattr(stage, "map_tracks", fake_map)
    comps, used, _, t = stage._map_pass(Path("db"), Path("im"), Path("w"), "p", 80, "global", 3)
    assert calls == ["global"] and used == "global" and [c.n for c in comps] == [43, 25, 6]  # 74/80 > 80 %
    assert t["mapping_s"] == 1.0


def test_fallback_keeps_the_mapper_that_registers_more_in_total(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    out = {"global": {0: _Rec(30), 1: _Rec(2)}, "incremental": {0: _Rec(28), 1: _Rec(20)}}

    def fake_map(db, images, path, *, mapper, verify=True):  # type: ignore[no-untyped-def]
        return out[mapper], {"verification_s": 0.1, "mapping_s": 2.0}

    monkeypatch.setattr(stage, "map_tracks", fake_map)
    comps, used, _, t = stage._map_pass(Path("db"), Path("im"), Path("w"), "p", 80, "global", 3)
    assert used == "incremental" and [c.n for c in comps] == [28, 20]
    assert t["mapping_s"] == 4.0
