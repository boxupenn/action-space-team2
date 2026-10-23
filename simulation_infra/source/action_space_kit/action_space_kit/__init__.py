# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments and UI extensions. These need a running Isaac Sim app (pxr/omni/carb);
# without one (e.g. `python -m action_space_kit.bridge.tello_bridge` on the ground station) the
# pure-Python parts of the kit (bridge, control, perception) must still be importable.
try:
    from .tasks import *
    from .ui_extension_example import *
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] not in ("pxr", "omni", "carb", "usdrt", "isaacsim"):
        raise
