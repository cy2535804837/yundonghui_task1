"""Shared YAML config loader for the grasp / place CLIs.

Both ``grasp_pose_grasp_execute`` and ``grasp_pose_place_execute`` ship a
``config.yaml`` next to their ``main.py``. That file holds every CLI parameter
so you can tune behaviour in ONE place instead of passing long command lines.

Keys are the argparse *dest* names: the long flag with the leading ``--`` removed
and dashes turned into underscores, e.g.::

    --grasp-tilt-y-deg 45   ->   grasp_tilt_y_deg: 45.0
    --no-release-on-finish  ->   release_on_finish: false
    --continuous-grasp-orientation -> continuous_grasp_orientation: true

Precedence (highest wins): explicit command-line flag > config.yaml > built-in
code default. Pass ``--config ''`` to ignore the file, or ``--write-config`` to
regenerate it from the current effective values.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

_WRITE_CONFIG_SENTINEL = "__USE_CONFIG_PATH__"


def default_config_path(module_file: str) -> str:
    """Path to the ``config.yaml`` bundled next to the given module file."""
    return os.path.join(os.path.dirname(os.path.abspath(module_file)), "config.yaml")


def add_config_args(p: argparse.ArgumentParser, default_path: str) -> None:
    """Add ``--config`` / ``--write-config`` to a parser."""
    p.add_argument(
        "--config",
        default=default_path,
        help="YAML file of CLI defaults (keys = argparse dests). Auto-loaded from "
        f"{default_path!r}. Pass --config '' to ignore it. Explicit CLI flags "
        "always override the file.",
    )
    p.add_argument(
        "--write-config",
        nargs="?",
        const=_WRITE_CONFIG_SENTINEL,
        default=None,
        metavar="PATH",
        help="Write the effective parameters (after --config + CLI overrides) to "
        "PATH (default: the --config path) as YAML, then exit. Use this to create "
        "or refresh the config file.",
    )


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML mapping from ``path`` (empty/missing -> {})."""
    if not path or not os.path.exists(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[CONFIG] {path!r} must be a YAML mapping (key: value)")
    return dict(data)


def apply_config_defaults(
    parser: argparse.ArgumentParser, argv: Optional[List[str]]
) -> Tuple[Dict[str, Any], str]:
    """Read ``--config`` from argv, load it and set it as parser defaults.

    Returns ``(applied, config_path)``. Call BEFORE ``parser.parse_args`` so the
    file's values become defaults that explicit CLI flags still override.
    """
    pre, _ = parser.parse_known_args(argv)
    path = getattr(pre, "config", "") or ""
    data = load_config(path)
    valid = {a.dest for a in parser._actions if a.dest != "help"}
    applied = {k: v for k, v in data.items() if k in valid}
    unknown = sorted(k for k in data if k not in valid)
    if unknown:
        print(f"[CONFIG] WARNING: {path}: ignoring unknown keys: {unknown}")
    if applied:
        parser.set_defaults(**applied)
        print(f"[CONFIG] loaded {len(applied)} params from {path}")
    return applied, path


def maybe_write_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_path: str,
) -> bool:
    """If ``--write-config`` was given, dump effective params and return True."""
    target = getattr(args, "write_config", None)
    if target is None:
        return False
    out_path = config_path if target == _WRITE_CONFIG_SENTINEL else str(target)
    dump_config(parser, args, out_path)
    return True


def dump_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace, out_path: str
) -> None:
    """Write every parameter (except meta flags) to ``out_path`` as YAML."""
    import yaml

    skip = {"help", "config", "write_config"}
    data = {
        a.dest: getattr(args, a.dest)
        for a in parser._actions
        if a.dest not in skip
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "# CLI parameters for this module. Keys are argparse dests (the long\n"
            "# flag with leading -- removed and dashes turned into underscores).\n"
            "# Edit values here instead of typing them on the command line.\n"
            "# Explicit command-line flags still override these values.\n"
            "# Regenerate with:  python3 -m <module>.main --write-config\n\n"
        )
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False, allow_unicode=True)
    print(f"[CONFIG] wrote {len(data)} params to {out_path}")
