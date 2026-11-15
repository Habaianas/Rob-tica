
#!/usr/bin/env python3
"""Connect to the Lite 6 and configure a supervised reduced-mode session."""

from __future__ import annotations

import argparse
import ipaddress
import math
import sys
from typing import Any, Optional

# ============================================================
# ROBOT CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.1.189"  # Replace with the Lite 6 IP address

MODE_POSITION = 0
MODE_MANUAL = 2

SET_STATE_MOTION = 0

# Safety caps from the project requirements (§3)
MAX_JOINT_SPEED_DEG_S = 54.0
MAX_TCP_SPEED_MM_S = 150.0

# Bench-approved TCP boundary
# Order: [x_max, x_min, y_max, y_min, z_max, z_min]
TCP_BOUNDARY_MM = [
    300.0,
    -300.0,
    300.0,
    -300.0,
    400.0,
    0.0
]

# Default settings used by log_joints.py
DEFAULT_CONFIG = None


# ============================================================
# VALIDATION
# ============================================================

def _validate_settings() -> tuple[str, float, float, list[float]]:

    if not isinstance(ROBOT_IP, str) or not ROBOT_IP.strip():
        raise ValueError("Set a valid ROBOT_IP in safe_session.py")

    try:
        ipaddress.ip_address(ROBOT_IP)
    except ValueError as exc:
        raise ValueError("ROBOT_IP must be a valid IP address") from exc

    joint_speed = float(MAX_JOINT_SPEED_DEG_S)
    tcp_speed = float(MAX_TCP_SPEED_MM_S)

    if not math.isfinite(joint_speed) or not 0 < joint_speed <= 54.0:
        raise ValueError("Invalid maximum joint speed")

    if not math.isfinite(tcp_speed) or not 0 < tcp_speed <= 150.0:
        raise ValueError("Invalid maximum TCP speed")

    if len(TCP_BOUNDARY_MM) != 6:
        raise ValueError("TCP boundary must contain six values")

    boundary = []

    for value in TCP_BOUNDARY_MM:
        if isinstance(value, bool):
            raise ValueError("TCP boundary values must be numeric")

        number = float(value)

        if not math.isfinite(number):
            raise ValueError("TCP boundary values must be finite")

        boundary.append(number)

    if any(boundary[i] <= boundary[i + 1] for i in (0, 2, 4)):
        raise ValueError("Invalid TCP boundary maximum/minimum")

    return ROBOT_IP.strip(), joint_speed, tcp_speed, boundary


def _check(code: Any, operation: str) -> None:

    if isinstance(code, bool) or not isinstance(code, int):
        raise RuntimeError(
            f"{operation} returned invalid status code {code!r}"
        )

    if code != 0:
        raise RuntimeError(
            f"{operation} failed with xArm status code {code}"
        )


# ============================================================
# SAFE SESSION
# ============================================================

def safe_session(
    config_path: Optional[Any] = None,
    *,
    mode: int = MODE_POSITION,
    api_factory: Optional[Any] = None,
) -> Any:
    """
    Connect to the Lite 6, apply reduced-mode safety settings,
    enable motion and return the XArmAPI object.

    mode=0: Position control
    mode=2: Supervised manual joint teaching
    """

    robot_ip, joint_speed, tcp_speed, boundary = _validate_settings()

    if mode not in (MODE_POSITION, MODE_MANUAL):
        raise ValueError("Mode must be 0 or 2")

    if api_factory is None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError(
                "xarm-python-sdk is missing. Install it using: "
                "python -m pip install xarm-python-sdk"
            ) from exc

        api_factory = XArmAPI

    arm = api_factory(robot_ip, is_radian=False)

    try:
        if getattr(arm, "connected", True) is False:
            raise RuntimeError(
                f"Could not connect to Lite 6 at {robot_ip}"
            )

        # Clear warnings and errors
        _check(arm.clean_warn(), "clean_warn")
        _check(arm.clean_error(), "clean_error")

        # Configure reduced-mode speed limits
        _check(
            arm.set_reduced_max_joint_speed(
                joint_speed,
                is_radian=False
            ),
            "set_reduced_max_joint_speed"
        )

        _check(
            arm.set_reduced_max_tcp_speed(tcp_speed),
            "set_reduced_max_tcp_speed"
        )

        # Configure TCP workspace boundary
        _check(
            arm.set_reduced_tcp_boundary(boundary),
            "set_reduced_tcp_boundary"
        )

        # Enable reduced mode
        _check(
            arm.set_reduced_mode(True),
            "set_reduced_mode"
        )

        # Enable motion and configure operating mode
        _check(
            arm.motion_enable(enable=True),
            "motion_enable"
        )

        _check(
            arm.set_mode(mode),
            "set_mode"
        )

        _check(
            arm.set_state(SET_STATE_MOTION),
            "set_state"
        )

        print("Lite 6 connected successfully.")
        print("Reduced mode enabled.")
        print(f"Maximum joint speed: {joint_speed} deg/s")
        print(f"Maximum TCP speed: {tcp_speed} mm/s")
        print(f"Operating mode: {mode}")

        return arm

    except Exception:

        try:
            _check(
                arm.motion_enable(enable=False),
                "motion_enable(False)"
            )
        except Exception as cleanup_error:
            print(
                f"Could not disable arm: {cleanup_error}",
                file=sys.stderr
            )

        try:
            arm.disconnect()
        except Exception as cleanup_error:
            print(
                f"Could not disconnect arm: {cleanup_error}",
                file=sys.stderr
            )

        raise


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    arm = None

    try:
        arm = safe_session(mode=MODE_POSITION)
        print("Safe session established.")
        return 0

    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Safe session failed: {exc}", file=sys.stderr)
        return 1

    finally:
        if arm is not None:

            try:
                _check(
                    arm.motion_enable(enable=False),
                    "motion_enable(False)"
                )
            except Exception as cleanup_error:
                print(
                    f"Could not disable arm: {cleanup_error}",
                    file=sys.stderr
                )

            try:
                arm.disconnect()
            except Exception as cleanup_error:
                print(
                    f"Could not disconnect arm: {cleanup_error}",
                    file=sys.stderr
                )


if __name__ == "__main__":
    raise SystemExit(main())
