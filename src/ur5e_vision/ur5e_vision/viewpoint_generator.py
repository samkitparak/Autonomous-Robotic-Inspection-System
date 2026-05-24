"""
viewpoint_generator.py
----------------------
Pure geometry — no ROS2 imports.  Generates end-effector waypoints for
orbiting around a detected object so the centered camera looks at the
object center from each angle.

Camera geometry (centered mount):
  - Camera is directly below tool0 (no lateral offset)
  - Camera optical axis = tool0 -Z axis (straight down relative to flange)

Derivation — for the camera to look at the object centre:
  1. Place the EE at (obj + r*cos θ, obj + r*sin θ, obj_z + h)
  2. Compute unit vector d from EE toward object center
  3. Set tool0 Z axis = d  (camera looks straight at object)
  4. Set tool0 X axis = orbit tangent, orthogonalized against d
  5. Derive quaternion from this rotation matrix (Shepperd's method)

  Height formula (30° elevation from horizontal):
    h = radius × tan(30°) ≈ 0.577 × radius
  (No camera-offset correction needed — camera is centered.)
"""

import math
from dataclasses import dataclass


_TAN30 = math.tan(math.radians(30))   # ≈ 0.5774 — 30° elevation ratio


def _mat_to_quat_xyzw(R: list[list[float]]) -> tuple[float, float, float, float]:
    """3×3 rotation matrix (list of rows) → (qx, qy, qz, qw). Shepperd's method."""
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    return (x, y, z, w)


@dataclass
class Viewpoint:
    """
    A single end-effector pose for the scan orbit.

    x, y, z     : EE position in robot base frame (metres)
    qx, qy, qz, qw : orientation quaternion — tool0 Z axis points toward object center
    """
    x:   float
    y:   float
    z:   float
    qx:  float = 0.0
    qy:  float = 0.0
    qz:  float = 0.0
    qw:  float = 1.0

    def to_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)


def _aim_at_quaternion(
    ex: float, ey: float, ez: float,
    obj_x: float, obj_y: float, obj_z: float,
    theta: float,
) -> tuple[float, float, float, float]:
    """
    Quaternion that orients tool0 so its Z axis points from (ex,ey,ez) toward
    (obj_x, obj_y, obj_z).  tool0 X is the orbit tangent at angle theta.
    Returns (qx, qy, qz, qw).
    """
    # tool0 Z: unit vector from EE toward object
    dx, dy, dz = obj_x - ex, obj_y - ey, obj_z - ez
    d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    zx, zy, zz = dx / d_len, dy / d_len, dz / d_len

    # tool0 X: orbit tangent at theta, then orthogonalized against Z
    tx, ty, tz = -math.sin(theta), math.cos(theta), 0.0
    dot = tx*zx + ty*zy + tz*zz
    tx -= dot * zx;  ty -= dot * zy;  tz -= dot * zz
    t_len = math.sqrt(tx*tx + ty*ty + tz*tz)
    xx, xy, xz = tx / t_len, ty / t_len, tz / t_len

    # tool0 Y: Z × X  (right-hand rule)
    yx = zy*xz - zz*xy
    yy = zz*xx - zx*xz
    yz = zx*xy - zy*xx

    # Rotation matrix: columns are tool0 axes expressed in world frame
    R = [
        [xx, yx, zx],
        [xy, yy, zy],
        [xz, yz, zz],
    ]
    return _mat_to_quat_xyzw(R)


def generate_orbit(
    obj_x:          float,
    obj_y:          float,
    obj_z:          float,
    radius:         float = 0.25,
    height:         float | None = None,
    num_points:     int   = 8,
    start_angle:    float = 0.0,
    sweep:          float = 2 * math.pi,
) -> list[Viewpoint]:
    """
    Generate EE viewpoints for an orbit around (obj_x, obj_y, obj_z).

    Args:
        obj_{x,y,z}  : object centre in base frame (metres)
        radius       : horizontal standoff from object centre (metres)
        height       : EE height above obj_z (metres).
                       None → h = radius × tan(30°)  (30° elevation)
        num_points   : number of waypoints
        start_angle  : angle of first waypoint from world +X axis (radians)
        sweep        : angular span (radians); 2π = full circle

    Returns:
        List of Viewpoint objects with aim-at orientations.
    """
    if height is None:
        height = radius * _TAN30

    height = max(height, 0.12)

    viewpoints: list[Viewpoint] = []
    for i in range(num_points):
        if num_points == 1:
            theta = start_angle
        elif sweep >= 2 * math.pi - 1e-6:
            theta = start_angle + sweep * i / num_points
        else:
            theta = start_angle + sweep * i / (num_points - 1)

        ex = obj_x + radius * math.cos(theta)
        ey = obj_y + radius * math.sin(theta)
        ez = obj_z + height

        qx, qy, qz, qw = _aim_at_quaternion(ex, ey, ez, obj_x, obj_y, obj_z, theta)
        viewpoints.append(Viewpoint(x=ex, y=ey, z=ez, qx=qx, qy=qy, qz=qz, qw=qw))

    return viewpoints


def generate_tiered_orbit(
    obj_x:       float,
    obj_y:       float,
    obj_z:       float,
    tiers:       list[tuple[float, int]] | None = None,
    start_angle: float = 0.0,
) -> list[Viewpoint]:
    """
    Generate a multi-ring orbit for complete multi-angle coverage.

    Each tier is (radius_m, num_points). Height is auto-computed per ring
    so the camera optical axis always aims at the object centre.

    Default tiers:
      - Inner  (r=0.20 m, 6 pts) — close-up for label reading
      - Middle (r=0.28 m, 8 pts) — main inspection ring
      - Outer  (r=0.35 m, 6 pts) — wide-angle overview

    Total: 20 viewpoints covering the object from every horizontal direction
    and three effective elevations (camera height scales with radius).
    """
    if tiers is None:
        tiers = [(0.20, 6), (0.28, 8), (0.35, 6)]

    viewpoints: list[Viewpoint] = []
    for radius, n_pts in tiers:
        ring = generate_orbit(
            obj_x, obj_y, obj_z,
            radius=radius,
            num_points=n_pts,
            start_angle=start_angle,
        )
        viewpoints.extend(ring)
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
    """
    dx = current_ee_x - obj_x
    dy = current_ee_y - obj_y
    return math.atan2(dy, dx)
