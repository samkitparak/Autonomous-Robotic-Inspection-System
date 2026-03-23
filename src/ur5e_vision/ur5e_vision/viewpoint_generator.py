"""
viewpoint_generator.py
----------------------
Pure geometry — no ROS2 imports.  Generates end-effector waypoints for
orbiting around a detected object so the offset wrist camera looks at the
object center from each angle.

Camera geometry (from hardware CAD):
  - Camera is 17 cm along tool0 +X from the flange
  - Camera optical axis is 60° from tool0 +Z (straight down) toward tool0 -X

Derivation — for the camera to look at the object centre:
  1. EE yaw = θ  →  tool0 X points AWAY from the object at angle θ
  2. Camera is then at  EE + 0.17·(cos θ, sin θ, 0)  in world frame
  3. Camera optical axis in world = (-sin60·cos θ, -sin60·sin θ, -cos60)
                                   = (-0.866 cos θ, -0.866 sin θ, -0.5)
  4. For the ray to pass through the object centre, the height must satisfy:
         h / (r + 0.17) = sin60 / cos60 = tan60/... = 0.5/0.866 = tan(30°) ≈ 0.577
     so  h_optimal = (r + CAMERA_OFFSET) × tan(30°)

Orientation — quaternion for tool0 pointing straight down with yaw ψ:
  Closed-form:  q = [w=0, x=cos(ψ/2), y=sin(ψ/2), z=0]
  Verified by:  R(q) = [[cosψ, sinψ, 0], [sinψ, -cosψ, 0], [0, 0, -1]]
    → tool0-Z = (0,0,-1) (pointing down ✓), tool0-X = (cosψ, sinψ, 0) (at angle ψ ✓)
"""

import math
from dataclasses import dataclass, field

# Physical camera constants (match camera_transform.launch.py)
CAMERA_OFFSET_M  = 0.17     # lateral offset from EE (metres)
CAMERA_PITCH_DEG = 60.0     # degrees from straight-down toward center

_TAN30 = math.tan(math.radians(30))   # ≈ 0.5774 — height/distance ratio


@dataclass
class Viewpoint:
    """
    A single end-effector pose for the scan orbit.

    x, y, z : position in robot base frame (metres)
    yaw     : tool0 yaw about world Z (radians).
              At this yaw, tool0 X points away from the object and the camera
              looks back toward the object center.
    """
    x:   float
    y:   float
    z:   float
    yaw: float   # radians

    def to_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        """
        Quaternion for tool0 pointing straight down with self.yaw rotation.
        Formula: q = [w=0, x=cos(yaw/2), y=sin(yaw/2), z=0]
        Returns (qx, qy, qz, qw).
        """
        half = self.yaw / 2.0
        return (math.cos(half), math.sin(half), 0.0, 0.0)


def generate_orbit(
    obj_x:          float,
    obj_y:          float,
    obj_z:          float,
    radius:         float = 0.25,
    height:         float | None = None,
    num_points:     int   = 8,
    start_angle:    float = 0.0,          # radians from world +X axis
    sweep:          float = 2 * math.pi,  # radians; 2π = full circle
) -> list[Viewpoint]:
    """
    Generate EE viewpoints for an orbit around (obj_x, obj_y, obj_z).

    Args:
        obj_{x,y,z}  : object centre in base frame (metres)
        radius       : horizontal standoff from object centre (metres)
        height       : EE height above obj_z (metres).
                       None → use optimal value  h = (radius + 0.17) × tan(30°)
        num_points   : number of waypoints (≥ 2 for a sweep, 1 for a single pose)
        start_angle  : angle of first waypoint from world +X axis (radians)
        sweep        : angular span of the orbit (radians).
                       2π = full circle, π = half circle, etc.

    Returns:
        List of Viewpoint objects, one per waypoint.
    """
    if height is None:
        height = (radius + CAMERA_OFFSET_M) * _TAN30   # optimal camera alignment

    height = max(height, 0.12)   # hard floor: never below 12 cm above object

    viewpoints: list[Viewpoint] = []
    for i in range(num_points):
        # Angular step: for full circle divide by num_points (wraps around);
        # for partial sweep divide by (num_points - 1) so endpoints are included.
        if num_points == 1:
            theta = start_angle
        elif sweep >= 2 * math.pi - 1e-6:   # full circle
            theta = start_angle + sweep * i / num_points
        else:
            theta = start_angle + sweep * i / (num_points - 1)

        ex = obj_x + radius * math.cos(theta)
        ey = obj_y + radius * math.sin(theta)
        ez = obj_z + height

        # EE yaw = theta so tool0 X points radially outward (away from object)
        viewpoints.append(Viewpoint(x=ex, y=ey, z=ez, yaw=theta))

    return viewpoints


def coverage_angles_for_label(
    current_ee_x: float,
    current_ee_y: float,
    obj_x:        float,
    obj_y:        float,
    num_points:   int = 6,
) -> float:
    """
    Return a start_angle for a half-orbit that begins from the current EE
    position (so the robot doesn't waste a move going to a far start point).
    Useful for calling generate_orbit with sweep=π.
    """
    dx = current_ee_x - obj_x
    dy = current_ee_y - obj_y
    return math.atan2(dy, dx)   # current EE angle from object
