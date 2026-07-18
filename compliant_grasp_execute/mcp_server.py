#!/usr/bin/env python3
"""
compliant_grasp_execute/mcp_server.py
=====================================

MCP server that exposes the single-arm **compliant** grasp (pick) phase as a
tool.  The compliant pipeline is the force/torque (F/T) admittance variant of
the grasp: the final descent to the object is done under F/T
admittance control (descend-to-contact) so the gripper stays compliant and
does not slam the object or the table.  It also supports the *elbow-high*
reconfiguration for awkward object poses.

The tool is **synchronous**: it kicks off the underlying pipeline
(``compliant_grasp_execute.main``) as a subprocess, blocks until it finishes,
and returns ``{"state": "succeed", ...}`` on a successful grasp or
``{"state": "failed", "msg": "..."}`` otherwise.

Why a subprocess (rather than calling ``main()`` in-process)?
* The pipeline calls ``rclpy.init()`` / ``rclpy.shutdown()`` and opens the
  gripper serial port, all of which must happen exactly once per Python
  process, so each call is isolated in a fresh subprocess.
* If anything in the pipeline crashes, only the subprocess dies -- the MCP
  server stays available for the next call.

A simple lock prevents two parallel invocations from fighting over the
shared hardware (both arms, the gripper, the camera).

Run
---
.. code-block:: bash

    # Streamable-HTTP on 0.0.0.0:8005/  (default)
    python3 -m compliant_grasp_execute.mcp_server

    # Or stdio (what Cursor / Claude Desktop's MCP config use)
    MCP_TRANSPORT=stdio python3 -m compliant_grasp_execute.mcp_server
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_LOG_DIR = _REPO_ROOT / "logs" / "mcp" / "compliant_grasp_execute"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_CONFIG = _PKG_DIR / "config.yaml"
_DEFAULT_HANDOFF = "/tmp/grasp_handoff.json"

# Serialise pipeline runs -- the arms / gripper / camera are owned
# exclusively by whoever is mid-run, so two parallel calls would crash both.
_run_lock = threading.Lock()

# Tracking + shutdown signalling for the currently-running subprocess.
_active_proc: Optional[subprocess.Popen] = None
_active_proc_lock = threading.Lock()
_shutdown_event = threading.Event()

# Linux-only: ask the kernel to SIGTERM the child if its parent (this MCP
# server) dies.  ``PR_SET_PDEATHSIG`` is constant 1 in <sys/prctl.h>.
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec hook: have the kernel SIGTERM the child if we die."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass  # not Linux / no libc -- best-effort only


def _kill_active_proc(grace: float = 3.0) -> None:
    """SIGTERM the active subprocess group, escalating to SIGKILL."""
    proc = _active_proc  # snapshot -- fine without the lock here
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
    deadline = time.monotonic() + max(0.0, float(grace))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


mcp = FastMCP("compliant_grasp_execute")


def _config_with_prompt(prompts: List[str]) -> str:
    """Write a temp config that is the package config.yaml with ``prompt``
    overridden.

    The compliant CLI's ``--prompt`` uses argparse ``append``, so passing
    ``--prompt X`` on the command line APPENDS to the config's default prompt
    list (e.g. ``[remote-control]`` -> ``[remote-control, X]``) instead of
    replacing it -- which would make detection search for the wrong object.
    To override cleanly we copy the whole config (preserving every tuned
    param) and set ``prompt`` to exactly the requested list, then pass it via
    ``--config`` (and do NOT pass ``--prompt``).
    """
    import yaml

    data: Dict[str, Any] = {}
    try:
        with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    data["prompt"] = list(prompts)
    fd, path = tempfile.mkstemp(prefix="compliant_grasp_cfg_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)
    return path


def _build_argv(
    *,
    prompt: str,
    arm: str,
    pipeline_version: Optional[str],
    motion_strategy: Optional[str],
    release_on_finish: Optional[bool],
    elbow_high: Optional[str],
    handoff_out: str,
    json_out: str,
    dry_run: bool,
    extra_args: Optional[List[str]],
    tmp_config_holder: List[str],
) -> List[str]:
    if arm not in ("auto", "left", "right"):
        raise ValueError(f"arm must be auto|left|right, got {arm!r}")

    argv: List[str] = ["--arm", arm, "--handoff-out", handoff_out, "--json-out", json_out]

    # Prompt override: because --prompt is append, a merged temp config is the
    # only clean way to REPLACE (not extend) the config's default object list.
    prompt = (prompt or "").strip()
    if prompt:
        cfg = _config_with_prompt([prompt])
        tmp_config_holder.append(cfg)
        argv += ["--config", cfg]
    # else: no --config / --prompt -> the package config.yaml default is used.

    if pipeline_version:
        argv += ["--pipeline-version", pipeline_version]
    if motion_strategy:
        argv += ["--motion-strategy", motion_strategy]
    # Only pass a flag when explicitly set, so config.yaml default applies
    # otherwise. For pick-then-place, pass release_on_finish=False to hold.
    if release_on_finish is not None:
        argv += ["--release-on-finish" if release_on_finish else "--no-release-on-finish"]
    # Elbow-high control. 'proactive' forces the top-grasp reconfiguration for
    # awkward (parallel-to-body) poses; 'off' disables it; None = config default.
    if elbow_high is not None:
        eh = str(elbow_high).lower()
        if eh in ("proactive", "on", "always"):
            argv += ["--elbow-high-proactive"]
            if eh == "always":
                argv += ["--elbow-high-always"]
        elif eh in ("off", "none", "disable", "disabled"):
            argv += ["--no-elbow-high-proactive", "--no-elbow-high-enable-fallback"]
        else:
            raise ValueError(
                f"elbow_high must be one of proactive|always|on|off, got {elbow_high!r}"
            )
    if dry_run:
        argv += ["--dry-run"]
    if extra_args:
        argv += list(extra_args)
    return argv


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@mcp.tool()
def grasp_object(
    prompt: str = "",
    arm: str = "auto",
    pipeline_version: Optional[str] = None,
    motion_strategy: Optional[str] = None,
    release_on_finish: Optional[bool] = None,
    elbow_high: Optional[str] = None,
    handoff_out: str = _DEFAULT_HANDOFF,
    dry_run: bool = False,
    timeout_sec: float = 600.0,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect an object by text prompt and execute a single-arm COMPLIANT grasp.

    The pipeline detects the object's pose, builds a TCP grasp pose, then moves
    the selected arm through approach -> compliant F/T descend-to-contact ->
    close gripper -> lift -> return home.  The final descent uses force/torque
    admittance control so the gripper stays compliant against the object and
    the table.  For awkward object poses (long axis parallel to the body) it can
    switch to an *elbow-high* top grasp.  This call **blocks** until the grasp
    finishes (typically tens of seconds) and returns its outcome.

    For a full pick-and-place, call this with ``release_on_finish=False``
    (keep holding) so the place phase can take over via the handoff file.

    Returns a ``dict`` of the form::

        {"state": "succeed", "msg": "", "result": {...}, "handoff_out": "...", ...}
        {"state": "failed",  "msg": "<reason>", "result": {...}, ...}

    Parameters
    ----------
    prompt:
        Object to grasp, e.g. ``"bottle"``, ``"banana"``, ``"remote-control"``.
        Empty (default) uses the object configured in ``config.yaml``.
    arm:
        ``auto`` | ``left`` | ``right`` (``auto`` picks the arm by object side).
    pipeline_version:
        Detection backend ``current`` | ``accelerated`` (``None`` = config default).
    motion_strategy:
        e.g. ``auto_hybrid``, ``auto_stream``, ``qp_all``, ``qp_stream``,
        ``qpik``, ``moveit`` (``None`` = config default).
    release_on_finish:
        If ``False``, keep holding the object (required for pick-and-place).
        ``None`` keeps the config default.
    elbow_high:
        ``proactive`` (force the elbow-high top grasp), ``always``, ``on``, or
        ``off`` (disable both proactive and fallback). ``None`` = config default.
    handoff_out:
        Path to write the grasp->place handoff JSON (default ``/tmp/grasp_handoff.json``).
    dry_run:
        Detect / plan only -- the robot does not move.
    timeout_sec:
        Maximum wall time before the subprocess is terminated and the call
        returns ``"failed"``.  Default 600 s.
    extra_args:
        Free-form extra CLI flags forwarded to ``compliant_grasp_execute.main``
        for any rarely-needed option not surfaced here, e.g.
        ``["--elbow-high-seed-tilt-up-deg", "12", "--grasp-tilt-y-deg", "45"]``.
    """
    global _active_proc

    if not _run_lock.acquire(blocking=False):
        return {
            "state": "failed",
            "msg": (
                "another grasp/place run is already active on this server; "
                "only one run at a time is allowed because the arms, gripper "
                "and camera are shared resources"
            ),
        }

    log_path = _LOG_DIR / (datetime.now().strftime("%Y%m%dT%H%M%S") + ".log")
    json_out = str(_LOG_DIR / (datetime.now().strftime("%Y%m%dT%H%M%S") + ".result.json"))
    started_at = time.time()
    tmp_configs: List[str] = []
    try:
        try:
            cli_argv = _build_argv(
                prompt=prompt,
                arm=arm,
                pipeline_version=pipeline_version,
                motion_strategy=motion_strategy,
                release_on_finish=release_on_finish,
                elbow_high=elbow_high,
                handoff_out=handoff_out,
                json_out=json_out,
                dry_run=dry_run,
                extra_args=extra_args,
                tmp_config_holder=tmp_configs,
            )
        except ValueError as e:
            return {"state": "failed", "msg": f"invalid argument: {e}"}

        cmd: List[str] = [
            sys.executable, "-u",
            "-m", "compliant_grasp_execute.main",
        ] + cli_argv

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc: Optional[subprocess.Popen] = None
        rc: Optional[int] = None
        shutdown_during_run = False
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_fh:
                log_fh.write(
                    f"[mcp] launched at {datetime.now().isoformat(timespec='seconds')}\n"
                    f"[mcp] cmd: {' '.join(cmd)}\n"
                    f"[mcp] cwd: {_REPO_ROOT}\n"
                    "[mcp] ----- begin subprocess output -----\n"
                )
                log_fh.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_REPO_ROOT),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    preexec_fn=_set_pdeathsig,
                )
                with _active_proc_lock:
                    _active_proc = proc

                # Poll-based wait so the SIGINT/SIGTERM handler can wake us
                # via ``_shutdown_event`` and we return promptly.
                deadline = time.monotonic() + float(timeout_sec)
                while True:
                    if _shutdown_event.is_set():
                        shutdown_during_run = True
                        rc = proc.poll()
                        break
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.2)

                if rc is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        rc = proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:  # noqa: BLE001
                            pass
                        rc = proc.wait()
                    msg = (
                        "server shutdown requested mid-run; subprocess terminated."
                        if shutdown_during_run
                        else f"timed out after {timeout_sec:.0f}s; subprocess terminated."
                    )
                    return {
                        "state": "failed",
                        "msg": msg + " See the log file for the partial trace.",
                        "exit_code": int(rc),
                        "duration_sec": round(time.time() - started_at, 2),
                        "log_path": str(log_path),
                    }
        except Exception as e:  # noqa: BLE001
            return {
                "state": "failed",
                "msg": f"failed to launch subprocess: {e!r}",
                "log_path": str(log_path),
            }
        finally:
            with _active_proc_lock:
                _active_proc = None

        duration = round(time.time() - started_at, 2)
        result = _read_json(json_out)

        if shutdown_during_run:
            return {
                "state": "failed",
                "msg": "server shutdown requested mid-run; subprocess terminated.",
                "exit_code": int(rc) if rc is not None else -1,
                "duration_sec": duration,
                "log_path": str(log_path),
                "result": result,
            }

        # The module always exits 0; success is encoded in the JSON "ok" field.
        ok = bool(result.get("ok")) if isinstance(result, dict) else False
        if rc == 0 and ok:
            return {
                "state": "succeed",
                "msg": "",
                "exit_code": 0,
                "duration_sec": duration,
                "log_path": str(log_path),
                "handoff_out": handoff_out,
                "result": result,
            }

        return {
            "state": "failed",
            "msg": _summarise_failure(log_path, exit_code=int(rc) if rc is not None else -1, result=result),
            "exit_code": int(rc) if rc is not None else -1,
            "duration_sec": duration,
            "log_path": str(log_path),
            "result": result,
        }
    finally:
        for cfg in tmp_configs:
            try:
                os.unlink(cfg)
            except OSError:
                pass
        _run_lock.release()


@mcp.tool()
def read_handoff(path: str = _DEFAULT_HANDOFF) -> Dict[str, Any]:
    """Read the grasp->place handoff JSON (which arm is holding what)."""
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    data = _read_json(path)
    if data is None:
        return {"exists": True, "path": path, "msg": "could not parse handoff JSON"}
    return {"exists": True, "path": path, "handoff": data}


def _summarise_failure(log_path: Path, *, exit_code: int, result: Optional[Dict[str, Any]] = None) -> str:
    """Produce a short, agent-friendly failure description.

    Prefers the structured result's failure flags; otherwise falls back to
    the last few non-empty lines of the log.
    """
    if isinstance(result, dict):
        motion = result.get("motion") or {}
        flags = {k: v for k, v in motion.items() if k.endswith("_ok")}
        if flags:
            return f"grasp did not fully succeed (ok={result.get('ok')}); motion flags: {flags}"

    tail: List[str] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines()]
        for ln in reversed(lines):
            if ln:
                tail.append(ln)
                if len(tail) >= 8:
                    break
    except OSError:
        pass
    tail.reverse()
    if not tail:
        return f"subprocess exited with code {exit_code} (see log at {log_path})"
    return f"subprocess exited with code {exit_code}; tail of log:\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_signal_count = 0


def _shutdown_handler(signum, _frame) -> None:
    global _signal_count
    _signal_count += 1
    name = signal.Signals(signum).name if signum in {s.value for s in signal.Signals} else str(signum)
    print(
        f"\n[mcp] caught {name} (count={_signal_count}); "
        + ("cleaning up..." if _signal_count == 1 else "FORCE EXIT"),
        file=sys.stderr,
        flush=True,
    )
    if _signal_count >= 2:
        os._exit(128 + signum)
    _shutdown_event.set()
    try:
        _kill_active_proc(grace=2.0)
    except Exception:  # noqa: BLE001
        pass
    os._exit(128 + signum)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Default: streamable-http on 0.0.0.0:8005 at path / (8003=grasp, 8004=place).
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=transport,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", "8005")),
            path=os.environ.get("MCP_PATH", "/"),
        )
