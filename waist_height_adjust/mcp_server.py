#!/usr/bin/env python3
"""
waist_height_adjust/mcp_server.py
=================================

MCP server that exposes the robot **waist (torso) height** adjustment as a tool.

The tianyi2 "body" is a 4-DOF chain
``[first_leg_pitch, second_leg_pitch, waist_pitch, waist_yaw]`` driven in
Cartesian space via ``ActionCall.endpose_body_controller`` with a 4-element
target ``[x, z, pitch, yaw]``.  ``z`` is the waist height (factory zero pose is
``[0.05, 0.68, 0.0, 0.0]``).  This tool sets that height while holding the other
body DOFs at safe defaults.

The tool is **synchronous**: it runs the underlying tool
(``waist_height_adjust.main``) as a subprocess, blocks until it finishes, and
returns ``{"state": "succeed", ...}`` on success or
``{"state": "failed", "msg": "..."}`` otherwise.

Why a subprocess (rather than calling ``main()`` in-process)?
* The tool calls ``rclpy.init()`` / ``rclpy.shutdown()``, which must happen
  exactly once per Python process, so each call is isolated in a fresh process.
* If anything crashes, only the subprocess dies -- the MCP server stays up.

A simple lock prevents two parallel invocations from fighting over the shared
body hardware.

Run
---
.. code-block:: bash

    # Streamable-HTTP on 0.0.0.0:8006/  (default; 8003=grasp, 8004=place,
    # 8005=compliant-grasp)
    python3 -m waist_height_adjust.mcp_server

    # Or stdio (what Cursor / Claude Desktop's MCP config use)
    MCP_TRANSPORT=stdio python3 -m waist_height_adjust.mcp_server
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_LOG_DIR = _REPO_ROOT / "logs" / "mcp" / "waist_height_adjust"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Serialise runs -- the body/leg hardware is owned exclusively by whoever is
# mid-run, so two parallel calls would fight over it.
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


mcp = FastMCP("waist_height_adjust")


def _build_argv(
    *,
    height: float,
    body_x: Optional[float],
    body_pitch: Optional[float],
    body_yaw: Optional[float],
    enable_hardware: Optional[bool],
    settle_sec: Optional[float],
    json_out: str,
    dry_run: bool,
    extra_args: Optional[List[str]],
) -> List[str]:
    argv: List[str] = ["--height", repr(float(height)), "--json-out", json_out]
    if body_x is not None:
        argv += ["--body-x", repr(float(body_x))]
    if body_pitch is not None:
        argv += ["--body-pitch", repr(float(body_pitch))]
    if body_yaw is not None:
        argv += ["--body-yaw", repr(float(body_yaw))]
    if enable_hardware is not None:
        argv += ["--enable-hardware" if enable_hardware else "--no-enable-hardware"]
    if settle_sec is not None:
        argv += ["--settle-sec", repr(float(settle_sec))]
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
def set_waist_height(
    height: float,
    body_x: Optional[float] = None,
    body_pitch: Optional[float] = None,
    body_yaw: Optional[float] = None,
    enable_hardware: Optional[bool] = None,
    settle_sec: Optional[float] = None,
    dry_run: bool = False,
    timeout_sec: float = 120.0,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Set the robot's waist (torso) HEIGHT to ``height`` metres.

    Drives the 4-DOF body ``[x, z, pitch, yaw]`` endpose controller, changing
    only the height (``z``) while holding the other body DOFs at their safe
    defaults, then reports the body joint angles before/after.  This call
    **blocks** until the move completes (a few seconds).

    Reachable range (tianyi2): the factory-tall / nominal height is ~0.68 m and
    the mechanical floor is ~0.56 m -- asking for a height below that makes a
    leg-pitch joint exceed its limit and the body controller REJECTS the goal
    (no motion).  In that case the returned dict has ``state="failed"`` and the
    ``result.hint`` explains it; try a height closer to 0.68 m.

    Returns a ``dict`` of the form::

        {"state": "succeed", "msg": "", "result": {...}, ...}
        {"state": "failed",  "msg": "<reason>", "result": {...}, ...}

    Parameters
    ----------
    height:
        Target waist height in metres (the body endpose ``z``). Reachable range
        is roughly ``[0.56, 0.68]`` m; the tool also clamps to its configured
        ``[min_height, max_height]`` safety bounds first.
    body_x:
        Body endpose forward offset ``x`` (m). ``None`` = config default
        (~0.05). Usually leave as-is.
    body_pitch:
        Body endpose pitch (rad). ``None`` = config default (0 = upright).
    body_yaw:
        Body endpose yaw (rad). ``None`` = config default (0 = facing forward).
    enable_hardware:
        Enable the leg + waist hardware before moving (needed on real hardware).
        ``None`` = config default (True).
    settle_sec:
        Seconds to wait after the move before reading back joint angles.
        ``None`` = config default.
    dry_run:
        Resolve/print the target only -- the robot does not move.
    timeout_sec:
        Max wall time before the subprocess is terminated and the call returns
        ``"failed"``.  Default 120 s.
    extra_args:
        Free-form extra CLI flags forwarded to ``waist_height_adjust.main`` for
        any rarely-needed option, e.g. ``["--min-height", "0.55"]``.
    """
    global _active_proc

    if not _run_lock.acquire(blocking=False):
        return {
            "state": "failed",
            "msg": (
                "another waist-height run is already active on this server; "
                "only one run at a time is allowed because the body hardware "
                "is a shared resource"
            ),
        }

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = _LOG_DIR / (stamp + ".log")
    json_out = str(_LOG_DIR / (stamp + ".result.json"))
    started_at = time.time()
    try:
        try:
            cli_argv = _build_argv(
                height=height,
                body_x=body_x,
                body_pitch=body_pitch,
                body_yaw=body_yaw,
                enable_hardware=enable_hardware,
                settle_sec=settle_sec,
                json_out=json_out,
                dry_run=dry_run,
                extra_args=extra_args,
            )
        except (ValueError, TypeError) as e:
            return {"state": "failed", "msg": f"invalid argument: {e}"}

        cmd: List[str] = [
            sys.executable, "-u",
            "-m", "waist_height_adjust.main",
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

                # Poll-based wait so the SIGINT/SIGTERM handler can wake us via
                # ``_shutdown_event`` and we return promptly.
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

        # The module exits 0 on success (ok=True) and 1 on failure; success is
        # also encoded in the JSON "ok" field.
        ok = bool(result.get("ok")) if isinstance(result, dict) else False
        if rc == 0 and ok:
            return {
                "state": "succeed",
                "msg": "",
                "exit_code": 0,
                "duration_sec": duration,
                "log_path": str(log_path),
                "result": result,
            }

        return {
            "state": "failed",
            "msg": _summarise_failure(
                log_path,
                exit_code=int(rc) if rc is not None else -1,
                result=result,
            ),
            "exit_code": int(rc) if rc is not None else -1,
            "duration_sec": duration,
            "log_path": str(log_path),
            "result": result,
        }
    finally:
        _run_lock.release()


def _summarise_failure(
    log_path: Path,
    *,
    exit_code: int,
    result: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a short, agent-friendly failure description.

    Prefers the structured result's ``hint`` (the tool already explains a
    controller rejection / out-of-range height); otherwise falls back to the
    last few non-empty lines of the log.
    """
    if isinstance(result, dict):
        hint = result.get("hint")
        if hint:
            return str(hint)

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

    # Default: streamable-http on 0.0.0.0:8006 at path / (8003=grasp,
    # 8004=place, 8005=compliant-grasp).
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=transport,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", "8006")),
            path=os.environ.get("MCP_PATH", "/"),
        )
