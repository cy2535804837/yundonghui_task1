#!/usr/bin/env python3
"""
grasp_place_mcp_server.py
=========================

A single MCP server that exposes BOTH the grasp (pick) and place phases as
tools on ONE port:

* ``grasp_object`` -> runs ``grasp_pose_grasp_execute.main``
* ``place_object`` -> runs ``grasp_pose_place_execute.main``
* ``read_handoff`` -> reads the grasp->place handoff JSON

This replaces running the two per-package servers (ports 8003 / 8004)
separately; both pick and place are served from the same process/port, just
with different tool names.

Each tool is **synchronous**: it launches the underlying pipeline as a
subprocess, blocks until it finishes, and returns ``{"state": "succeed", ...}``
on success or ``{"state": "failed", "msg": "..."}`` otherwise.

Why a subprocess (rather than calling ``main()`` in-process)?
* Each pipeline calls ``rclpy.init()`` / ``rclpy.shutdown()`` and opens the
  gripper serial port, all of which must happen exactly once per process, so
  every call is isolated in a fresh subprocess.
* If anything crashes, only the subprocess dies — the server stays up.

A single lock serialises ALL runs (grasp AND place) because the arms, gripper
and camera are shared hardware: only one pipeline may run at a time.

Run
---
.. code-block:: bash

    # Streamable-HTTP on 0.0.0.0:8005/  (default)
    python3 grasp_place_mcp_server.py

    # Or stdio (what Cursor / Claude Desktop's MCP config use)
    MCP_TRANSPORT=stdio python3 grasp_place_mcp_server.py
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

_REPO_ROOT = Path(__file__).resolve().parent
_LOG_ROOT = _REPO_ROOT / "logs" / "mcp"
_LOG_ROOT.mkdir(parents=True, exist_ok=True)

_DEFAULT_HANDOFF = "/tmp/grasp_handoff.json"

# Serialise ALL pipeline runs (grasp + place): the arms / gripper / camera are
# owned exclusively by whoever is mid-run, so two parallel calls would crash.
_run_lock = threading.Lock()

# Tracking + shutdown signalling for the currently-running subprocess.
_active_proc: Optional[subprocess.Popen] = None
_active_proc_lock = threading.Lock()
_shutdown_event = threading.Event()

# Linux-only: ask the kernel to SIGTERM the child if its parent (this server)
# dies. ``PR_SET_PDEATHSIG`` is constant 1 in <sys/prctl.h>.
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec hook: have the kernel SIGTERM the child if we die."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass  # not Linux / no libc — best-effort only


def _kill_active_proc(grace: float = 3.0) -> None:
    """SIGTERM the active subprocess group, escalating to SIGKILL."""
    proc = _active_proc  # snapshot — fine without the lock here
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


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _summarise_failure(
    log_path: Path,
    *,
    phase: str,
    exit_code: int,
    result: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a short, agent-friendly failure description.

    Prefers the structured result's failure flags; otherwise falls back to the
    last few non-empty lines of the log.
    """
    if isinstance(result, dict):
        motion = result.get("motion") or {}
        flags = {k: v for k, v in motion.items() if k.endswith("_ok")}
        if flags:
            return (
                f"{phase} did not fully succeed (ok={result.get('ok')}); "
                f"motion flags: {flags}"
            )

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


def _run_pipeline(
    *,
    module: str,
    phase: str,
    log_subdir: str,
    cli_argv: List[str],
    timeout_sec: float,
    extra_success: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Launch ``python -m <module> <cli_argv> --json-out <tmp>`` and wait.

    Shared by both ``grasp_object`` and ``place_object``. Returns the standard
    ``{"state": "succeed"|"failed", ...}`` dict.
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

    log_dir = _LOG_ROOT / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / (ts + ".log")
    json_out = str(log_dir / (ts + ".result.json"))
    started_at = time.time()
    try:
        cmd: List[str] = [
            sys.executable, "-u", "-m", module,
            *cli_argv, "--json-out", json_out,
        ]

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc: Optional[subprocess.Popen] = None
        rc: Optional[int] = None
        shutdown_during_run = False
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_fh:
                log_fh.write(
                    f"[mcp] launched at {datetime.now().isoformat(timespec='seconds')}\n"
                    f"[mcp] phase: {phase}\n"
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

        # The modules always exit 0; success is encoded in the JSON "ok" field.
        ok = bool(result.get("ok")) if isinstance(result, dict) else False
        if rc == 0 and ok:
            out = {
                "state": "succeed",
                "msg": "",
                "exit_code": 0,
                "duration_sec": duration,
                "log_path": str(log_path),
                "result": result,
            }
            if extra_success:
                out.update(extra_success)
            return out

        return {
            "state": "failed",
            "msg": _summarise_failure(
                log_path,
                phase=phase,
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


mcp = FastMCP("grasp_place_execute")


# ---------------------------------------------------------------------------
# Tool: grasp_object  (pick)
# ---------------------------------------------------------------------------
def _build_grasp_argv(
    *,
    prompt: str,
    arm: str,
    pipeline_version: Optional[str],
    motion_strategy: Optional[str],
    release_on_finish: Optional[bool],
    handoff_out: str,
    dry_run: bool,
    extra_args: Optional[List[str]],
) -> List[str]:
    if not prompt:
        raise ValueError("prompt is required (the object to grasp, e.g. 'bottle')")
    if arm not in ("auto", "left", "right"):
        raise ValueError(f"arm must be auto|left|right, got {arm!r}")

    argv: List[str] = ["--prompt", prompt, "--arm", arm, "--handoff-out", handoff_out]
    if pipeline_version:
        argv += ["--pipeline-version", pipeline_version]
    if motion_strategy:
        argv += ["--motion-strategy", motion_strategy]
    # Only pass a flag when explicitly set, so config.yaml default applies
    # otherwise. For pick-then-place, pass release_on_finish=False to hold.
    if release_on_finish is not None:
        argv += ["--release-on-finish" if release_on_finish else "--no-release-on-finish"]
    if dry_run:
        argv += ["--dry-run"]
    if extra_args:
        argv += list(extra_args)
    return argv


@mcp.tool()
def grasp_object(
    prompt: str = "bottle",
    arm: str = "auto",
    pipeline_version: Optional[str] = None,
    motion_strategy: Optional[str] = None,
    release_on_finish: Optional[bool] = None,
    handoff_out: str = _DEFAULT_HANDOFF,
    dry_run: bool = False,
    timeout_sec: float = 600.0,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect an object by text prompt and execute a single-arm grasp.

    The pipeline detects the object's pose, builds a TCP grasp pose, then moves
    the selected arm through approach -> grasp (close gripper) -> lift -> return
    home. This call **blocks** until the grasp finishes (typically tens of
    seconds) and returns its outcome.

    For a full pick-and-place, call this with ``release_on_finish=False`` (keep
    holding) so ``place_object`` can take over via the handoff file.

    Returns a ``dict`` of the form::

        {"state": "succeed", "msg": "", "result": {...}, "handoff_out": "...", ...}
        {"state": "failed",  "msg": "<reason>", "result": {...}, ...}

    Parameters
    ----------
    prompt:
        Object to grasp, e.g. ``"bottle"``, ``"banana"``, ``"sponge"``.
    arm:
        ``auto`` | ``left`` | ``right`` (``auto`` picks the arm by object side).
    pipeline_version:
        Detection backend ``current`` | ``accelerated`` (``None`` = config default).
    motion_strategy:
        e.g. ``qp_all``, ``auto_hybrid``, ``moveit``, ``qpik``, ``qp_stream``
        (``None`` = config default).
    release_on_finish:
        If ``False``, keep holding the object (required for pick-and-place).
        ``None`` keeps the config default.
    handoff_out:
        Path to write the grasp->place handoff JSON (default ``/tmp/grasp_handoff.json``).
    dry_run:
        Detect / plan only — the robot does not move.
    timeout_sec:
        Maximum wall time before the subprocess is terminated and the call
        returns ``"failed"``. Default 600 s.
    extra_args:
        Free-form extra CLI flags forwarded to ``grasp_pose_grasp_execute.main``
        for any rarely-needed option not surfaced here,
        e.g. ``["--grasp-tilt-y-deg", "15", "--approach-dist", "0.12"]``.
    """
    try:
        cli_argv = _build_grasp_argv(
            prompt=prompt,
            arm=arm,
            pipeline_version=pipeline_version,
            motion_strategy=motion_strategy,
            release_on_finish=release_on_finish,
            handoff_out=handoff_out,
            dry_run=dry_run,
            extra_args=extra_args,
        )
    except ValueError as e:
        return {"state": "failed", "msg": f"invalid argument: {e}"}

    return _run_pipeline(
        module="grasp_pose_grasp_execute.main",
        phase="grasp",
        log_subdir="grasp_execute",
        cli_argv=cli_argv,
        timeout_sec=timeout_sec,
        extra_success={"handoff_out": handoff_out},
    )


# ---------------------------------------------------------------------------
# Tool: place_object  (place)
# ---------------------------------------------------------------------------
def _build_place_argv(
    *,
    place_x: Optional[float],
    place_y: Optional[float],
    place_z: Optional[float],
    place_z_clearance: Optional[float],
    place_tilt_y_deg: Optional[float],
    arm: str,
    motion_strategy: Optional[str],
    require_holding: Optional[bool],
    handoff_in: str,
    dry_run: bool,
    extra_args: Optional[List[str]],
) -> List[str]:
    if arm not in ("auto", "left", "right"):
        raise ValueError(f"arm must be auto|left|right, got {arm!r}")

    # Explicit target: require all three coordinates together, else use the
    # handoff grasp position (place back where it was picked).
    have = [v for v in (place_x, place_y, place_z) if v is not None]
    if have and len(have) != 3:
        raise ValueError("place_x, place_y and place_z must all be provided together (or none)")

    argv: List[str] = ["--handoff-in", handoff_in, "--arm", arm]
    if len(have) == 3:
        argv += ["--place-x", f"{float(place_x)}",
                 "--place-y", f"{float(place_y)}",
                 "--place-z", f"{float(place_z)}"]
    if place_z_clearance is not None:
        argv += ["--place-z-clearance", f"{float(place_z_clearance)}"]
    if place_tilt_y_deg is not None:
        argv += ["--place-tilt-y-deg", f"{float(place_tilt_y_deg)}"]
    if motion_strategy:
        argv += ["--motion-strategy", motion_strategy]
    if require_holding is not None:
        argv += ["--require-holding" if require_holding else "--no-require-holding"]
    if dry_run:
        argv += ["--dry-run"]
    if extra_args:
        argv += list(extra_args)
    return argv


@mcp.tool()
def place_object(
    place_x: Optional[float] = None,
    place_y: Optional[float] = None,
    place_z: Optional[float] = None,
    place_z_clearance: Optional[float] = None,
    place_tilt_y_deg: Optional[float] = None,
    arm: str = "auto",
    motion_strategy: Optional[str] = None,
    require_holding: Optional[bool] = None,
    handoff_in: str = _DEFAULT_HANDOFF,
    dry_run: bool = False,
    timeout_sec: float = 600.0,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Place the object the robot is currently holding, then release it.

    Reads the grasp handoff to learn which arm is holding the object, moves it
    through approach -> place -> open gripper (release) -> lift -> return home.
    This call **blocks** until the place finishes and returns its outcome.

    If ``place_x/place_y/place_z`` are omitted, the object is placed back where
    it was grasped (from the handoff ``grasp_pose7``); otherwise pass all three
    for an explicit TCP target in ``waist_yaw_link`` (metres).

    Returns a ``dict`` of the form::

        {"state": "succeed", "msg": "", "result": {...}, ...}
        {"state": "failed",  "msg": "<reason>", "result": {...}, ...}

    Parameters
    ----------
    place_x, place_y, place_z:
        Target TCP position in ``waist_yaw_link`` (m). Provide all three or none
        (none = place back at the grasp pose).
    place_z_clearance:
        Release this far above the target Z (m). ``None`` = config default (~0.05).
    place_tilt_y_deg:
        Nose-down tilt for the placement orientation (deg).
    arm:
        ``auto`` | ``left`` | ``right`` (``auto`` uses the holding arm from the handoff).
    motion_strategy:
        e.g. ``qp_all``, ``auto_hybrid`` (``None`` = config default).
    require_holding:
        If ``False``, place even when the handoff says nothing is held.
    handoff_in:
        Path to the grasp->place handoff JSON (default ``/tmp/grasp_handoff.json``).
    dry_run:
        Plan only — the robot does not move.
    timeout_sec:
        Maximum wall time before the subprocess is terminated. Default 600 s.
    extra_args:
        Free-form extra CLI flags forwarded to ``grasp_pose_place_execute.main``.
    """
    try:
        cli_argv = _build_place_argv(
            place_x=place_x,
            place_y=place_y,
            place_z=place_z,
            place_z_clearance=place_z_clearance,
            place_tilt_y_deg=place_tilt_y_deg,
            arm=arm,
            motion_strategy=motion_strategy,
            require_holding=require_holding,
            handoff_in=handoff_in,
            dry_run=dry_run,
            extra_args=extra_args,
        )
    except ValueError as e:
        return {"state": "failed", "msg": f"invalid argument: {e}"}

    return _run_pipeline(
        module="grasp_pose_place_execute.main",
        phase="place",
        log_subdir="place_execute",
        cli_argv=cli_argv,
        timeout_sec=timeout_sec,
    )


# ---------------------------------------------------------------------------
# Tool: read_handoff
# ---------------------------------------------------------------------------
@mcp.tool()
def read_handoff(path: str = _DEFAULT_HANDOFF) -> Dict[str, Any]:
    """Read the grasp->place handoff JSON (which arm is holding what)."""
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    data = _read_json(path)
    if data is None:
        return {"exists": True, "path": path, "msg": "could not parse handoff JSON"}
    return {"exists": True, "path": path, "handoff": data}


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

    # Default: streamable-http on 0.0.0.0:8005 at path /. Both grasp_object and
    # place_object are served here; pick the tool by name.
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
