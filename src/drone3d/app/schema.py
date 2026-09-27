"""The configuration as a UI schema: every tunable option, its type, default and help.

Built from the dataclasses in :mod:`drone3d.config` and the comments next to
their fields in its source, so the form can never drift from the code: a new
option appears in the UI with its explanation as soon as it exists.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import typing
from pathlib import Path
from typing import Any

__all__ = ["config_schema", "profiles"]

_FIELD = re.compile(r"^    (\w+):\s*([^=#]+?)\s*(?:=\s*(.+?))?\s*(?:#\s*(.*))?$")


def _comments(source: str) -> dict[str, dict[str, str]]:
    """``{class name: {field: help}}`` from trailing comments and the comment block above a field."""
    out: dict[str, dict[str, str]] = {}
    cls, pending = None, []
    for line in source.splitlines():
        m = re.match(r"^class (\w+)", line)
        if m:
            cls, pending = m.group(1), []
            out[cls] = {}
            continue
        if cls is None:
            continue
        stripped = line.strip()
        if stripped.startswith("#") and line.startswith("    ") and not line.startswith("        "):
            pending.append(stripped.lstrip("#").strip())
            continue
        f = _FIELD.match(line)
        if f and not stripped.startswith(("def ", "return", "@")):
            help_text = " ".join(pending + ([f.group(4)] if f.group(4) else [])).strip()
            out[cls][f.group(1)] = help_text
            pending = []
        elif stripped and not stripped.startswith(('"""', "'''")):
            pending = []
    return out


def _kind(hint: Any) -> tuple[str, bool]:
    """-> (``int|float|bool|str|list[int]|list[str]``, optional)."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    optional = False
    if (
        origin in (typing.Union, getattr(__import__("types"), "UnionType", None))
        and type(None) in args
    ):
        optional = True
        hint = next(a for a in args if a is not type(None))
        origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is list:
        inner = args[0] if args else str
        return f"list[{getattr(inner, '__name__', 'str')}]", optional
    return getattr(hint, "__name__", "str"), optional


def _choices(help_text: str, kind: str) -> list[str] | None:
    if kind != "str" or "|" not in help_text:
        return None
    head = (
        help_text.split(";")[0]
        if help_text.count("|") < help_text.split(";")[0].count("|") + 1
        else help_text
    )
    parts = [re.match(r"\s*([\w.-]+)", p) for p in head.split("|")]
    names = [p.group(1) for p in parts if p]
    return names if len(names) >= 2 else None


def config_schema() -> dict:
    """``{"top": [fields], "sections": [{"key", "title", "fields": [...]}], "stages": [...]}``."""
    from drone3d import config as C

    helps = _comments(inspect.getsource(C))
    pipeline_hints = typing.get_type_hints(C.PipelineConfig)
    defaults = C.PipelineConfig()
    top, sections = [], []
    for f in dataclasses.fields(C.PipelineConfig):
        hint = pipeline_hints[f.name]
        if dataclasses.is_dataclass(hint):
            hints = typing.get_type_hints(hint)
            sub = getattr(defaults, f.name)
            fields = []
            for g in dataclasses.fields(hint):
                kind, optional = _kind(hints[g.name])
                help_text = helps.get(hint.__name__, {}).get(g.name, "")
                fields.append({"key": g.name, "type": kind, "optional": optional, "default": getattr(sub, g.name),
                               "choices": _choices(help_text, kind), "help": help_text})  # fmt: skip
            doc = (inspect.getdoc(hint) or "").split("\n")[0]
            sections.append({"key": f.name, "title": doc or f.name, "fields": fields})
        else:
            kind, optional = _kind(hint)
            help_text = helps.get("PipelineConfig", {}).get(f.name, "")
            top.append({"key": f.name, "type": kind, "optional": optional, "default": getattr(defaults, f.name),
                        "choices": _choices(help_text, kind), "help": help_text})  # fmt: skip
    return {"top": top, "sections": sections, "stages": list(C.ALL_STAGES)}


def profiles(config_dir: Path) -> list[dict]:
    """Each ``configs/*.yaml``: name, the header comment as description, and its values."""
    import yaml

    out = []
    for path in sorted(config_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        header = []
        for line in text.splitlines():
            if not line.startswith("#"):
                break
            header.append(line.lstrip("#").strip())
        # the header's prose, not its command-line examples
        desc = " ".join(
            h
            for h in header
            if h and not h.startswith(("uv run", "--", "[", "drone3d run", "drone3d ui", "drone3d view")) and "--set" not in h
        ).strip()
        out.append(
            {
                "name": path.stem,
                "path": str(path),
                "description": desc,
                "values": yaml.safe_load(text) or {},
            }
        )
    return out
