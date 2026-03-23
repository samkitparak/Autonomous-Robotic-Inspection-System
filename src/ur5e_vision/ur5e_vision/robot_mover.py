"""
robot_mover.py
--------------
Thin wrapper around MoveIt2 MoveGroup action for end-effector position control.

All public methods are blocking and thread-safe — they return only after the
motion completes or fails.  Designed to be called from a background planning
thread alongside a MultiThreadedExecutor.

Requires: MoveIt2 running with /move_action server and /compute_ik service.
"""

import math
import threading

import rclpy
import rclpy.time
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    MotionPlanRequest,
    PlanningOptions,
)
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory

from ur5e_vision.viewpoint_generator import Viewpoint


class RobotMover:
    """
    Executes end-effector position moves via MoveIt2 MoveGroup action.

    Strategy:
      1. Call /compute_ik with the current joint state as seed to get the
         IK solution closest to the current configuration (prevents wrap-around).
      2. Plan in joint space to that specific configuration (no pose-based IK
         inside OMPL, so no wrap-around possible at planning time either).
      3. Publish the planned trajectory directly to the UR controller topic
         (bypasses MoveIt2's execute layer which mangles timestamps).
    """

    PLANNING_GROUP = 'ur_manipulator'
    EE_LINK        = 'tool0'
    BASE_FRAME     = 'base_link'

    # UR5e joint names (same order as /joint_states)
    ARM_JOINTS = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]

    def __init__(self, node: Node, speed: float = 0.15):
        self.node  = node
        self.speed = speed

        self._client   = ActionClient(node, MoveGroup, '/move_action')
        self._ik_client = node.create_client(GetPositionIK, '/compute_ik')

        # Publish planned trajectories directly to the UR controller topic.
        # This bypasses MoveIt2's execution layer (which mangles timestamps)
        # and is the same approach used in sanity.py for this robot.
        self._traj_pub = node.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

        # Track current joint state so we can seed MoveIt2 planning correctly.
        self._joint_state      = None
        self._joint_state_lock = threading.Lock()
        node.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)

        if not self._client.wait_for_server(timeout_sec=10.0):
            node.get_logger().warn(
                '/move_action server not available — is MoveIt2 running? '
                'Moves will fail until the server comes up.')

        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().warn(
                '/compute_ik service not available — is MoveIt2 running?')

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def move_to_pose(self, x: float, y: float, z: float) -> tuple[bool, str]:
        """
        Move the end-effector to (x, y, z) in base_link frame.
        The current orientation is preserved (robot keeps how it is tilted).
        Returns (success, human-readable message).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.BASE_FRAME, self.EE_LINK,
                rclpy.time.Time(), timeout=Duration(seconds=1.0))
            orientation = tf.transform.rotation
        except Exception as e:
            return False, f'Cannot read current orientation via TF: {e}'

        target = PoseStamped()
        target.header.frame_id  = self.BASE_FRAME
        target.header.stamp     = self.node.get_clock().now().to_msg()
        target.pose.position.x  = float(x)
        target.pose.position.y  = float(y)
        target.pose.position.z  = float(z)
        target.pose.orientation = orientation

        return self._send_pose_goal(target)

    def move_to_viewpoint(self, vp: Viewpoint) -> tuple[bool, str]:
        """
        Move to a Viewpoint with its computed orientation (pointing down + yaw).
        Used by the scan_orbit executor so each waypoint has the correct heading.
        """
        qx, qy, qz, qw = vp.to_quaternion_xyzw()

        target = PoseStamped()
        target.header.frame_id  = self.BASE_FRAME
        target.header.stamp     = self.node.get_clock().now().to_msg()
        target.pose.position.x  = float(vp.x)
        target.pose.position.y  = float(vp.y)
        target.pose.position.z  = float(vp.z)
        target.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        return self._send_pose_goal(target)

    def execute_orbit(
        self,
        viewpoints: list[Viewpoint],
        on_waypoint_reached=None,
    ) -> tuple[int, int]:
        """
        Execute a list of Viewpoints in sequence, each via MoveIt2.

        Args:
            viewpoints          : ordered list from viewpoint_generator
            on_waypoint_reached : optional callable(index, viewpoint) called
                                  after each successful move (e.g. to capture image)
        Returns:
            (num_succeeded, num_total)
        """
        succeeded = 0
        for i, vp in enumerate(viewpoints):
            self.node.get_logger().info(
                f'Orbit waypoint {i + 1}/{len(viewpoints)}: '
                f'({vp.x:.3f}, {vp.y:.3f}, {vp.z:.3f})  yaw={math.degrees(vp.yaw):.0f}°')

            ok, msg = self.move_to_viewpoint(vp)
            if ok:
                succeeded += 1
                if on_waypoint_reached:
                    on_waypoint_reached(i, vp)
            else:
                self.node.get_logger().warn(
                    f'Orbit waypoint {i + 1} failed: {msg} — skipping.')

        return succeeded, len(viewpoints)

    def get_current_ee_pose(self) -> dict | None:
        """
        Returns current end-effector position as {'x', 'y', 'z'} in base frame.
        Returns None if TF is unavailable.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.BASE_FRAME, self.EE_LINK,
                rclpy.time.Time(), timeout=Duration(seconds=1.0))
            t = tf.transform.translation
            return {'x': t.x, 'y': t.y, 'z': t.z}
        except Exception as e:
            self.node.get_logger().warn(
                f'get_current_ee_pose: TF error: {e}', throttle_duration_sec=5)
            return None

    def _joint_state_cb(self, msg: JointState):
        with self._joint_state_lock:
            self._joint_state = msg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_pose_goal(self, target: PoseStamped) -> tuple[bool, str]:
        """
        Plan and execute a move to the given target pose.

        Two-step process:
          1. /compute_ik with current joints as seed → nearby IK solution
          2. Joint-space plan from current → IK solution (no wrap-around)
        """
        with self._joint_state_lock:
            js = self._joint_state

        # ----------------------------------------------------------------
        # Step 1: Compute IK seeded with current joint state
        # ----------------------------------------------------------------
        # Seeding with the current state forces KDL/TRAC-IK to return the
        # solution closest to where the robot already is, instead of a
        # wrap-around solution (e.g. shoulder_pan going -330° instead of +30°).
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name       = self.PLANNING_GROUP
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.pose_stamped     = target
        if js is not None:
            ik_req.ik_request.robot_state.joint_state = js

        ik_result = self._call_service_sync(self._ik_client, ik_req, timeout_sec=5.0)
        if ik_result is None:
            return False, 'IK service timeout'

        ik_code = ik_result.error_code.val
        if ik_code != MoveItErrorCodes.SUCCESS:
            code_map = {v: k for k, v in vars(MoveItErrorCodes).items()
                        if isinstance(v, int)}
            return False, f'IK failed: {code_map.get(ik_code, str(ik_code))}'

        goal_js = ik_result.solution.joint_state  # nearby joint configuration

        # Log and sanity-check the IK solution
        if js is not None:
            cur  = dict(zip(js.name,      js.position))
            goal = dict(zip(goal_js.name, goal_js.position))
            deltas = {n: abs(cur.get(n, 0) - goal.get(n, 0)) for n in self.ARM_JOINTS}
            max_delta = max(deltas.values())
            self.node.get_logger().info(
                f'IK solution: max_delta={max_delta:.3f} rad  '
                f'pan: {cur.get("shoulder_pan_joint", 0):.3f}'
                f'→{goal.get("shoulder_pan_joint", 0):.3f}')
            if max_delta > math.pi:
                return False, (
                    f'IK solution still wraps around ({max_delta:.2f} rad on '
                    f'{max(deltas, key=deltas.get)}) — target may be unreachable nearby')

        # ----------------------------------------------------------------
        # Step 2: Joint-space motion plan to the IK solution
        # ----------------------------------------------------------------
        req = MotionPlanRequest()
        req.group_name                      = self.PLANNING_GROUP
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = float(self.speed)
        req.max_acceleration_scaling_factor = float(self.speed) * 0.5

        if js is not None:
            req.start_state.joint_state = js
        else:
            req.start_state.is_diff = True

        # Goal: the specific joint configuration returned by IK (tolerance ±1°)
        goal_positions = dict(zip(goal_js.name, goal_js.position))
        goal_c = Constraints()
        tol    = math.radians(1.0)
        for name in self.ARM_JOINTS:
            if name not in goal_positions:
                continue
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = goal_positions[name]
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight          = 1.0
            goal_c.joint_constraints.append(jc)
        req.goal_constraints = [goal_c]

        # ---- Plan only ----
        plan_goal = MoveGroup.Goal()
        plan_goal.request          = req
        plan_goal.planning_options = PlanningOptions(plan_only=True)

        plan_result = self._call_action_sync(plan_goal, timeout_sec=15.0)
        if plan_result is None:
            return False, 'Planning timed out or goal was rejected'

        code = plan_result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_map = {v: k for k, v in vars(MoveItErrorCodes).items()
                        if isinstance(v, int)}
            return False, f'MoveIt2 error: {code_map.get(code, str(code))}'

        traj = plan_result.result.planned_trajectory.joint_trajectory
        if not traj.points:
            return False, 'Planner returned empty trajectory'

        duration = (traj.points[-1].time_from_start.sec
                    + traj.points[-1].time_from_start.nanosec / 1e9)

        # Diagnostic: log first few waypoints to confirm sensible timing
        def _t(pt):
            return pt.time_from_start.sec + pt.time_from_start.nanosec / 1e9

        self.node.get_logger().info(
            f'Plan: {len(traj.points)} pts, {duration:.1f}s')
        for i, pt in enumerate(traj.points[:3]):
            self.node.get_logger().info(
                f'  pt[{i}] t={_t(pt):.4f}s  '
                f'pos={[round(p, 4) for p in pt.positions]}')
        if len(traj.points) > 3:
            pt = traj.points[-1]
            self.node.get_logger().info(
                f'  pt[-1] t={_t(pt):.4f}s  '
                f'pos={[round(p, 4) for p in pt.positions]}')

        # Abort if trajectory start doesn't match current robot state
        with self._joint_state_lock:
            js_now = self._joint_state
        if js_now is not None:
            cur = dict(zip(js_now.name, js_now.position))
            pt0 = dict(zip(traj.joint_names, traj.points[0].positions))
            max_err = max(abs(cur.get(j, 0) - pt0.get(j, 0)) for j in traj.joint_names)
            self.node.get_logger().info(
                f'Start-state check: max error = {max_err:.4f} rad')
            if max_err > 0.15:
                self.node.get_logger().error(
                    f'Trajectory start mismatch ({max_err:.3f} rad) — aborting.')
                return False, f'Start-state mismatch ({max_err:.3f} rad)'

        # ---- Execute: publish directly to the controller topic ----
        # stamp=0 → "start executing from now" (ROS convention)
        traj.header.stamp.sec     = 0
        traj.header.stamp.nanosec = 0
        self._traj_pub.publish(traj)

        # ---- Wait for completion via joint-state monitoring ----
        import time as _time
        goal_joints = dict(zip(traj.joint_names, traj.points[-1].positions))
        # 0.15 rad tolerance (~8.6°): UR controller stops slightly before goal
        # due to velocity scaling.  Extra 8s buffer covers slow final approach.
        deadline = _time.monotonic() + duration + 8.0
        while _time.monotonic() < deadline:
            _time.sleep(0.05)
            with self._joint_state_lock:
                js = self._joint_state
            if js is None:
                continue
            current = dict(zip(js.name, js.position))
            if all(abs(current.get(j, 999) - goal_joints[j]) < 0.15
                   for j in traj.joint_names):
                return True, 'Success'

        return False, 'Execution timed out (robot may not have reached goal)'

    def _call_service_sync(self, client, request, timeout_sec: float = 10.0):
        """Call a ROS2 service synchronously from a background thread."""
        result_box = [None]
        done       = threading.Event()

        def _on_result(future):
            try:
                result_box[0] = future.result()
            except Exception as e:
                self.node.get_logger().error(f'Service call failed: {e}')
            done.set()

        client.call_async(request).add_done_callback(_on_result)
        done.wait(timeout=timeout_sec)
        return result_box[0]

    def _call_action_sync(self, goal, timeout_sec: float = 30.0):
        """
        Send an action goal and block until completion.
        Safe to call from a background thread with MultiThreadedExecutor:
        the executor processes callbacks in its own threads; we just wait
        on a threading.Event.
        """
        result_box = [None]
        done       = threading.Event()

        def _on_result(future):
            result_box[0] = future.result()
            done.set()

        def _on_goal_response(future):
            handle = future.result()
            if not handle or not handle.accepted:
                done.set()
                return
            handle.get_result_async().add_done_callback(_on_result)

        self._client.send_goal_async(goal).add_done_callback(_on_goal_response)
        done.wait(timeout=timeout_sec)
        return result_box[0]
