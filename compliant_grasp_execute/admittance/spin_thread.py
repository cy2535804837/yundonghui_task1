"""
adaptive_place_right/spin_thread.py
===================================
Tiny background ``rclpy.spin_once`` helper.

Extracted verbatim from ``handover.pipeline._SpinThread`` so this
placement folder does not need to import the full (and heavy) handover
pipeline module just for one utility class.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy


class _SpinThread:
    """Run rclpy.spin_once on a single node in a background thread.

    Used while the main thread is busy orchestrating hardware.  Stopped
    before blocking action calls so ``spin_until_future_complete`` can own
    the node again.
    """

    def __init__(self, node, spin_timeout_sec: float = 0.01):
        self._node = node
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timeout = spin_timeout_sec

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PlacementSpin"
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            try:
                rclpy.spin_once(self._node, timeout_sec=self._timeout)
            except Exception as e:
                print(f"[ADP-R] spin error (continuing): {e}")
                time.sleep(0.05)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def is_spinning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = ["_SpinThread"]
