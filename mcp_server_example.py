#!/usr/bin/env python3
"""
bottle_cup_pour_place/mcp_server.py
====================================

MCP server that exposes the dual-arm bottle-cup pour-place pipeline
(with optional mobile-base navigation around the cup handover) as a
single tool.

The tool is **synchronous**: it kicks off the underlying pipeline
(``bottle_cup_pour_place.detect_pour_place_with_mobile``) as a
subprocess, blocks until the pipeline finishes, and returns
``{"state": "succeed", ...}`` on a clean exit or
``{"state": "failed", "msg": "..."}`` otherwise.

Why a subprocess (rather than calling ``main()`` in-process)?
* The pipeline calls ``rclpy.init()``, opens the dexterous-hand serial
  port, and spawns its own children (``adaptive_place``).  All of those
  must come up exactly once per Python process, so each call is isolated
  in a fresh subprocess.
* If anything in the pipeline crashes, only the subprocess dies — the
  MCP server stays available for the next call.

A simple lock prevents two parallel invocations from fighting over the
shared hardware (both arms, both hands, the mobile base, the camera).

Run
---
.. code-block:: bash

    # Streamable-HTTP on 0.0.0.0:8002/  (default — what curl / browsers / Cursor expect)
    python3 -m bottle_cup_pour_place.mcp_server

    # Or stdio (what Cursor / Claude Desktop's MCP config use)
    MCP_TRANSPORT=stdio python3 -m bottle_cup_pour_place.mcp_server
"""
from __future__ import annotations

import ctypes
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs" / "mcp" / "pour_place_mobile"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Serialise pipeline runs — the underlying hardware (both arms, both
# hands, the mobile base, the camera) is owned exclusively by whoever
# is mid-run, so two parallel calls would crash both.
_run_lock = threading.Lock()

# Tracking + shutdown signalling for the currently-running pipeline
# subprocess.  Used by the SIGINT/SIGTERM handlers in ``__main__`` so
# that a Ctrl+C on the server tears down both the server *and* the
# in-flight pipeline (otherwise uvicorn waits indefinitely for the
# blocked tool call, and the pipeline subprocess would be orphaned in
# its own session).
_active_proc: Optional[subprocess.Popen] = None
_active_proc_lock = threading.Lock()
_shutdown_event = threading.Event()


# Linux-only: ask the kernel to send the subprocess a SIGTERM if its
# parent (this MCP server) dies.  ``preexec_fn`` runs in the child after
# fork() but before exec(), so it's the safest place to install this.
# ``PR_SET_PDEATHSIG`` is constant 1 in <sys/prctl.h>.
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec hook: have the kernel SIGTERM the child if we die."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass  # not Linux / no libc — best-effort only


def _kill_active_proc(grace: float = 3.0) -> None:
    """SIGTERM the active pipeline subprocess group, escalating to SIGKILL.

    Safe to call from a signal handler (no Python locks acquired
    that could deadlock).
    """
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


mcp = FastMCP("pour_place_mobile")


def _build_argv(
    *,
    left_prompt: str,
    right_prompt: str,
    poses_json: Optional[str],
    replay_pour_json: Optional[str],
    replay_pour_rate_scale: Optional[float],
    mobile_loc1_xyt: Optional[List[float]],
    mobile_loc2_xyt: Optional[List[float]],
    skip_mobile: bool,
    skip_mobile_loc1: bool,
    skip_mobile_loc2: bool,
    dry_run: bool,
    extra_args: Optional[List[str]],
) -> List[str]:
    argv: List[str] = ["--left-prompt", left_prompt, "--right-prompt", right_prompt]
    if poses_json:
        argv += ["--poses-json", poses_json]
    if replay_pour_json:
        argv += ["--replay-pour-json", replay_pour_json]
    if replay_pour_rate_scale is not None:
        argv += ["--replay-pour-rate-scale", f"{float(replay_pour_rate_scale):.4f}"]

    def _xyt(v: List[float]) -> str:
        if len(v) != 3:
            raise ValueError(f"location must be [x, y, yaw] (3 floats), got {v!r}")
        return f"{float(v[0])},{float(v[1])},{float(v[2])}"

    if mobile_loc1_xyt is not None:
        argv += ["--mobile-loc1", _xyt(mobile_loc1_xyt)]
    if mobile_loc2_xyt is not None:
        argv += ["--mobile-loc2", _xyt(mobile_loc2_xyt)]

    if skip_mobile:
        argv += ["--skip-mobile"]
    if skip_mobile_loc1:
        argv += ["--skip-mobile-loc1"]
    if skip_mobile_loc2:
        argv += ["--skip-mobile-loc2"]
    if dry_run:
        argv += ["--dry-run"]

    if extra_args:
        argv += list(extra_args)
    return argv


@mcp.tool()
def pour_place_with_mobile(
    left_prompt: str = "bottle",
    right_prompt: str = "cup",
    poses_json: Optional[str] = "pour.json",
    replay_pour_json: Optional[str] = "bimanual_pour_teach_dv2.json",
    replay_pour_rate_scale: Optional[float] = None,
    mobile_loc1_xyt: Optional[List[float]] = None,
    mobile_loc2_xyt: Optional[List[float]] = None,
    skip_mobile: bool = False,
    skip_mobile_loc1: bool = False,
    skip_mobile_loc2: bool = False,
    dry_run: bool = False,
    timeout_sec: float = 600.0,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the dual-arm bottle-cup pour-place pipeline (with optional mobile-base nav).

    The pipeline grasps a bottle (left) and a cup (right) at fixed
    calibrated waist-frame poses, pours the bottle into the cup,
    optionally drives the mobile base to a delivery location for
    handover, opens the right gripper after a person takes the cup, and
    returns the arms / base home.  This call **blocks** until the whole
    pipeline finishes (typical duration: 60-180 s end-to-end), and
    returns its outcome.

    Returns a ``dict`` of the form::

        {"state": "succeed", "msg": "", "log_path": "...", "duration_sec": 87.4}
        {"state": "failed",  "msg": "<reason>", "log_path": "...", ...}

    Parameters
    ----------
    left_prompt, right_prompt:
        Object names (kept for future re-detection support, but the
        pipeline currently uses fixed calibrated centroids and ignores
        them).
    poses_json:
        Path (relative to the repo root) to the recorded approach /
        middle / placement poses JSON.  Default ``pour.json``.
    replay_pour_json:
        Path to the recorded bimanual pour trajectory JSON.  ``None``
        falls back to the built-in joint-space pour.  Default
        ``bimanual_pour_teach_dv2.json``.
    replay_pour_rate_scale:
        Pour playback speed multiplier (1.0 = recorded speed, >1 faster).
        ``None`` keeps the script default (1.5).
    mobile_loc1_xyt:
        ``[x, y, yaw]`` (metres, radians) in the SLAM map frame for the
        cup-handover location.  ``None`` keeps the script default.
    mobile_loc2_xyt:
        Final parking location after the handover.  ``None`` keeps the
        script default.
    skip_mobile:
        Disable mobile-base nav entirely (behave like
        ``detect_pour_place``).
    skip_mobile_loc1, skip_mobile_loc2:
        Skip just the navigation to ``loc1`` / ``loc2`` individually.
    dry_run:
        Plan-and-print mode: arms / mobile base do not move.
    timeout_sec:
        Maximum wall time the call will wait before terminating the
        subprocess and returning ``"failed"``.  Default 600 s.
    extra_args:
        Free-form extra CLI flags forwarded to
        ``detect_pour_place_with_mobile`` for any rarely-needed option
        not surfaced explicitly here.
    """
    global _active_proc

    if not _run_lock.acquire(blocking=False):
        return {
            "state": "failed",
            "msg": (
                "another pour-place run is already active on this server; "
                "only one run at a time is allowed because both arms, both "
                "hands, the mobile base, and the camera are shared resources"
            ),
        }

    log_path = _LOG_DIR / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + ".log"
    )
    started_at = time.time()
    try:
        try:
            cli_argv = _build_argv(
                left_prompt=left_prompt,
                right_prompt=right_prompt,
                poses_json=poses_json,
                replay_pour_json=replay_pour_json,
                replay_pour_rate_scale=replay_pour_rate_scale,
                mobile_loc1_xyt=mobile_loc1_xyt,
                mobile_loc2_xyt=mobile_loc2_xyt,
                skip_mobile=skip_mobile,
                skip_mobile_loc1=skip_mobile_loc1,
                skip_mobile_loc2=skip_mobile_loc2,
                dry_run=dry_run,
                extra_args=extra_args,
            )
        except ValueError as e:
            return {"state": "failed", "msg": f"invalid argument: {e}"}

        cmd: List[str] = [
            sys.executable, "-u",
            "-m", "bottle_cup_pour_place.detect_pour_place_with_mobile",
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

                # Poll-based wait (instead of blocking ``proc.wait``) so
                # the SIGINT/SIGTERM handler in __main__ can wake us up
                # via ``_shutdown_event`` and we can return promptly,
                # which in turn lets uvicorn finish its graceful
                # shutdown instead of hanging on this in-flight request.
                deadline = time.monotonic() + float(timeout_sec)
                while True:
                    # Check shutdown first so a Ctrl+C-driven kill is
                    # reported as "server shutdown" rather than the
                    # generic exit-code-summary.
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
                    # Either timeout or shutdown requested — terminate
                    # the whole subprocess group cleanly, escalating to
                    # SIGKILL if it ignores SIGTERM.
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
                    if shutdown_during_run:
                        return {
                            "state": "failed",
                            "msg": (
                                "server shutdown requested mid-run; "
                                "subprocess terminated. See the log "
                                "file for the partial trace."
                            ),
                            "exit_code": int(rc),
                            "duration_sec": round(time.time() - started_at, 2),
                            "log_path": str(log_path),
                        }
                    return {
                        "state": "failed",
                        "msg": (
                            f"timed out after {timeout_sec:.0f}s; the "
                            "subprocess was terminated. See the log file "
                            "for the partial trace."
                        ),
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
        if shutdown_during_run:
            # Shutdown was requested mid-run AND the subprocess had
            # already exited (presumably from the SIGTERM dispatched by
            # the signal handler) — report it as a shutdown rather than
            # a generic failure.
            return {
                "state": "failed",
                "msg": (
                    "server shutdown requested mid-run; subprocess "
                    "terminated. See the log file for the partial trace."
                ),
                "exit_code": int(rc) if rc is not None else -1,
                "duration_sec": duration,
                "log_path": str(log_path),
            }
        if rc == 0:
            return {
                "state": "succeed",
                "msg": "",
                "exit_code": 0,
                "duration_sec": duration,
                "log_path": str(log_path),
            }

        return {
            "state": "failed",
            "msg": _summarise_failure(log_path, exit_code=int(rc) if rc is not None else -1),
            "exit_code": int(rc) if rc is not None else -1,
            "duration_sec": duration,
            "log_path": str(log_path),
        }
    finally:
        _run_lock.release()


def _summarise_failure(log_path: Path, *, exit_code: int) -> str:
    """Produce a short, agent-friendly failure description.

    Includes the exit code and the last few non-empty lines of the log
    so the agent gets some signal about what went wrong without having
    to fetch the whole file.
    """
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
    return (
        f"subprocess exited with code {exit_code}; tail of log:\n"
        + "\n".join(tail)
    )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
# Counter so a *second* Ctrl+C is a hard kill — the first one tries a
# clean shutdown (kill the active pipeline subprocess group, then exit),
# but if anything is still wedged, mashing Ctrl+C again exits the
# server unconditionally.
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

    # Wake the polling tool loop so it returns promptly.
    _shutdown_event.set()
    # Best-effort terminate the active pipeline subprocess group.
    try:
        _kill_active_proc(grace=2.0)
    except Exception:  # noqa: BLE001
        pass
    # Now exit.  ``os._exit`` skips Python finalisers (which is what we
    # want — uvicorn's graceful-shutdown path can otherwise deadlock on
    # already-released resources).
    os._exit(128 + signum)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Install our signal handlers BEFORE starting the server.  Uvicorn
    # may install its own SIGINT handler when ``mcp.run()`` boots, but
    # SIGTERM typically remains ours, and even if uvicorn replaces both
    # the *second* Ctrl+C will trip Python's default-SIGINT behaviour
    # (KeyboardInterrupt → process exit), at which point
    # PR_SET_PDEATHSIG on the pipeline subprocess ensures it dies too.
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Default: streamable-http on 0.0.0.0:8002 at path /.
    # Override with env vars (or just edit the call).
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=transport,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", "8002")),
            path=os.environ.get("MCP_PATH", "/"),
        )
