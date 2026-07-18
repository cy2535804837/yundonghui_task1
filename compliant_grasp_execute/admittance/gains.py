"""
compliant_grasp_execute/admittance/gains.py
============================================
Admittance gain + force-processing dataclasses and the default F/T
calibration paths for this self-contained project.

Extracted (and trimmed) from ft_place_right/handover/config.py so the
compliant-grasp project carries no cross-dependency on ft_place_right.

``AdmittanceGains`` mirrors the structure consumed by
``AdmittanceController_v4_2_fixed.TranslationAdmittanceFixed`` /
``RotationAdmittanceFixed`` (per-axis M / B / K lists) so a single axis can
be made soft (insertion) while the others stay stiff (lateral) — exactly
what the compliant grasp insert needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# F/T calibration JSONs produced by ``ft_calibration/calibrate_ft.py`` and
# loaded at runtime by AdmittanceArm. They live inside this project so it is
# fully self-contained. The current robot is a NEW machine, so BOTH must be
# (re)generated here before the compliant grasp can be trusted.
_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FT_CALIB_DIR = os.path.join(_PROJECT_DIR, "ft_calibration")
DEFAULT_LEFT_FT_CALIB = os.path.join(_FT_CALIB_DIR, "ft_calibration_left.json")
DEFAULT_RIGHT_FT_CALIB = os.path.join(_FT_CALIB_DIR, "ft_calibration_right.json")


@dataclass
class AdmittanceGains:
    """Second-order admittance parameters for one arm (translation + rotation).

    M / B / K are per-axis (3-element) lists in the waist frame:
        M * a + B * v + K * (p - p_ref) = F
    """

    M: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    B: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    K: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M_rot: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    B_rot: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    K_rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max_vel: float = 20.0
    max_omega: float = 20.0
    rot_lead_time: float = 0.15


@dataclass
class ForceProcessing:
    force_deadzone: float = 0.5
    torque_deadzone: float = 0.05
    force_threshold: float = 0.5
    torque_threshold: float = 0.15
    filter_alpha: float = 0.8
    calib_samples: int = 200


DEFAULT_FORCE_PROC = ForceProcessing()
