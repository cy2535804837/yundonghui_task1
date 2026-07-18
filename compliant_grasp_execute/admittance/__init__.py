"""
compliant_grasp_execute.admittance
===================================
Self-contained admittance-control stack for the compliant grasp insert.

Verbatim copies of the proven math (AdmittanceController_v3 /
AdmittanceController_v4_2_fixed) plus the threaded ``AdmittanceArm`` runner,
re-packaged here so the grasp project does not depend on ft_place_right.
"""

from .gains import (
    AdmittanceGains,
    ForceProcessing,
    DEFAULT_FORCE_PROC,
    DEFAULT_LEFT_FT_CALIB,
    DEFAULT_RIGHT_FT_CALIB,
)
from .admittance_arm import AdmittanceArm, TFHelper
from .spin_thread import _SpinThread

__all__ = [
    "AdmittanceGains",
    "ForceProcessing",
    "DEFAULT_FORCE_PROC",
    "DEFAULT_LEFT_FT_CALIB",
    "DEFAULT_RIGHT_FT_CALIB",
    "AdmittanceArm",
    "TFHelper",
    "_SpinThread",
]
