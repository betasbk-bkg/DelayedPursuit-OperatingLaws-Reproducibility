#!/usr/bin/env python3
"""
Run ONE (path, v, tau) point of the Nav2 validation sweep and write its
cross-track RMSE to JSON.

Sequence
  1. publish /initialpose at the path start, with the heading of the first
     segment -- the loopback simulator ignores cmd_vel until it has one
  2. wait for the map->base_link transform to appear
  3. send the whole multi-lap polyline to controller_server's /follow_path action
  4. sample map->base_link while it runs
  5. compute cross-track RMSE over the steady-state window only

EXCLUSION WINDOW (plan v2 §2.3, and why it is not optional)
  - the first `settle_laps` laps are the start transient: the robot sits still for
    tau, then converges
  - the last `tail_exclude_m` metres are dropped because RPP's
    approachVelocityConstraint is applied unconditionally -- it is not behind any
    boolean, so it cannot be switched off, and it decelerates the robot inside
    approach_velocity_scaling_dist (0.6 m) of the path end
  Samples are assigned to the window by their matched path index, not by time, so
  the window is the same physical stretch of path at every speed.

Params (ROS): path_kind, side, radius, laps, n_sides, spacing, desired_vel,
              tau, settle_laps, tail_exclude_m, sample_hz, out_file, timeout_scale
"""
import json
import math
import os
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from letg2_nav2_validation import path_gen


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class RunPoint(Node):

    def __init__(self):
        super().__init__('letg2_run_point')
        d = self.declare_parameter
        d('path_kind', 'square'); d('side', 5.0); d('radius', 5.0)
        d('laps', 4.75); d('n_sides', 4); d('spacing', 0.05)
        d('desired_vel', 0.82); d('tau', 0.2)
        d('settle_laps', 1.5); d('tail_exclude_m', 3.5)   # >= path_gen.EXIT_M
        d('sample_hz', 50.0); d('timeout_scale', 2.0)
        d('out_file', '/tmp/letg2/point.json')

        g = lambda k: self.get_parameter(k).value
        self.kind = g('path_kind')
        self.vel = float(g('desired_vel')); self.tau = float(g('tau'))
        self.laps = float(g('laps')); self.spacing = float(g('spacing'))
        self.settle_laps = float(g('settle_laps'))
        self.tail_m = float(g('tail_exclude_m'))
        self.out_file = g('out_file')
        self.timeout_scale = float(g('timeout_scale'))

        if self.kind == 'circle':
            self.pts = path_gen.circle_path(float(g('radius')), self.laps, self.spacing)
            self.lap_len = 2 * math.pi * float(g('radius'))
        else:
            n = int(g('n_sides'))
            self.pts = path_gen.ngon_path(n, float(g('side')), self.laps, self.spacing)
            self.lap_len = n * float(g('side'))
        self.total_len = path_gen.path_length(self.pts)

        # The goal checker latches as soon as the robot is within xy_goal_tolerance of
        # the FINAL pose.  A whole number of laps ends where it started, so the goal is
        # satisfied at t=0 and FollowPath returns SUCCEEDED immediately with no motion.
        # Hence fractional laps (default 4.75): the path ends three quarters of the way
        # round, far from the start.  Guard against reintroducing the bug.
        d0 = math.hypot(self.pts[-1][0] - self.pts[0][0], self.pts[-1][1] - self.pts[0][1])
        if d0 < 1.0:
            self.get_logger().error(
                f'path ends {d0:.2f} m from its start: the goal checker will fire '
                f'immediately and the run will return SUCCEEDED with no motion. '
                f'Use fractional laps (e.g. {int(self.laps)}.75).')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.ac = ActionClient(self, FollowPath, 'follow_path')

        self.samples = []          # (t, x, y, matched_index)
        self.last_idx = 0
        self.sample_dt = 1.0 / float(g('sample_hz'))

        # Achieved delay from the delay node ([requested, mean, sd, min, max, n]).
        # Kept as the latest message; written into the result so the analysis can
        # use the delay actually applied rather than the one requested.
        from std_msgs.msg import Float64MultiArray
        self.delay_stats = None
        self.create_subscription(Float64MultiArray, '/letg2/delay_stats',
                                 self._on_delay_stats, 10)

    def _on_delay_stats(self, msg):
        if len(msg.data) >= 6:
            self.delay_stats = list(msg.data)

    # ---------------------------------------------------------------- helpers
    def make_path_msg(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, (x, y) in enumerate(self.pts):
            nx, ny = self.pts[min(i + 1, len(self.pts) - 1)]
            yaw = math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else 0.0
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            q = yaw_to_quat(yaw)
            (ps.pose.orientation.x, ps.pose.orientation.y,
             ps.pose.orientation.z, ps.pose.orientation.w) = q
            msg.poses.append(ps)
        return msg

    def publish_initial_pose(self, tol=0.05, timeout=20.0):
        """
        Teleport the robot to the path start -- and CONFIRM it happened.

        This must be verified, not assumed.  Each sweep point runs in a fresh
        process, so the /initialpose publisher may not have finished discovery
        when the first messages go out; they are then silently dropped and the
        robot stays wherever the previous point left it.  When the previous path
        ended at the same place as this one's goal, FollowPath returns SUCCEEDED
        instantly with no motion and the point reports NaN -- which is exactly how
        this failed the first time.

        So: wait for a subscriber, publish, then poll map->base_link until it
        actually reaches the requested pose.
        """
        x0, y0 = self.pts[0]
        x1, y1 = self.pts[1]
        yaw = math.atan2(y1 - y0, x1 - x0)

        t0 = time.time()
        while self.init_pub.get_subscription_count() < 1 and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.init_pub.get_subscription_count() < 1:
            self.get_logger().error('nobody subscribes to /initialpose')
            return False

        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        q = yaw_to_quat(yaw)
        m.pose.pose.position.x, m.pose.pose.position.y = float(x0), float(y0)
        (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
         m.pose.pose.orientation.z, m.pose.pose.orientation.w) = q

        while time.time() - t0 < timeout:
            m.header.stamp = self.get_clock().now().to_msg()
            self.init_pub.publish(m)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.03)
            try:
                tr = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(),
                                                     Duration(seconds=0.2))
                dx = tr.transform.translation.x - x0
                dy = tr.transform.translation.y - y0
                if math.hypot(dx, dy) <= tol:
                    self.get_logger().info(
                        f'initial pose confirmed ({x0:.2f}, {y0:.2f}) yaw {yaw:.3f} '
                        f'after {time.time()-t0:.1f}s')
                    return True
            except Exception:
                pass
        self.get_logger().error(
            f'robot did not reach the start pose ({x0:.2f}, {y0:.2f}) within {timeout}s')
        return False

    def wait_for_tf(self, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(),
                                            Duration(seconds=0.2)):
                return True
        return False

    def sample_pose(self, t_start):
        try:
            tr = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(),
                                                 Duration(seconds=0.1))
        except Exception:
            return
        x = tr.transform.translation.x
        y = tr.transform.translation.y
        _, idx = path_gen.cross_track((x, y), self.pts, self.last_idx, window=300)
        self.last_idx = idx
        self.samples.append((time.time() - t_start, x, y, idx))

    # ------------------------------------------------------------------- main
    def execute(self):
        # publish_initial_pose() both bootstraps and confirms: the loopback
        # simulator publishes NO map->base_link at all until it has received its
        # first /initialpose, so waiting for the transform before sending one
        # deadlocks (that is exactly what rc=2 meant on a freshly launched stack).
        # Its loop therefore publishes and polls TF together.
        if not self.publish_initial_pose():
            return 5
        if not self.ac.wait_for_server(timeout_sec=20.0):
            self.get_logger().error('follow_path action server not available')
            return 3

        goal = FollowPath.Goal()
        goal.path = self.make_path_msg()
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = 'general_goal_checker'

        self.get_logger().info(
            f'{self.kind}: {len(self.pts)} poses, {self.total_len:.1f} m, '
            f'v={self.vel:.3f}, tau={self.tau:.3f}')

        send = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        gh = send.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('goal rejected')
            return 4

        result_fut = gh.get_result_async()
        budget = self.total_len / max(self.vel, 1e-3) * self.timeout_scale + self.tau + 30.0
        t0 = time.time()
        next_s = 0.0
        while rclpy.ok() and not result_fut.done():
            rclpy.spin_once(self, timeout_sec=0.005)
            el = time.time() - t0
            if el >= next_s:
                self.sample_pose(t0)
                next_s += self.sample_dt
            if el > budget:
                self.get_logger().warn(f'timeout after {el:.0f}s; cancelling')
                gh.cancel_goal_async()
                break
        status = result_fut.result().status if result_fut.done() else GoalStatus.STATUS_UNKNOWN
        self.finish(status, time.time() - t0)
        return 0

    def finish(self, status, elapsed):
        lo_arc = self.settle_laps * self.lap_len
        hi_arc = max(self.total_len - self.tail_m, 0.0)
        lo_idx = path_gen.arc_index(self.pts, lo_arc)
        hi_idx = path_gen.arc_index(self.pts, hi_arc)

        # The window above spans a NON-INTEGER number of laps (1.5 -> 3.75 laps of
        # a 3.9-lap run is 2.225 laps), so it ends part-way through a corner cycle.
        # E7's five identical rows showed why that matters: runs differing by one
        # or two control periods in length -- the goal check fires once per control
        # cycle, so the run length is quantised -- land the window's far edge at a
        # different phase of that partial corner and the RMSE moves by about 2%.
        # That is the clustered, non-Gaussian scatter E7 measured (u* sd 1.52%).
        #
        # An INTEGER number of laps has no partial corner, so the phase cannot
        # matter.  Both are reported: rmse_m keeps the original definition so this
        # run stays comparable with every block already measured, and
        # rmse_m_intlap is the phase-insensitive one.  Adopt the latter only after
        # a replicate block shows it actually reduces the scatter.
        n_int = int(math.floor((hi_arc - lo_arc) / self.lap_len + 1e-9))
        hi_arc_int = lo_arc + n_int * self.lap_len
        hi_idx_int = path_gen.arc_index(self.pts, hi_arc_int)

        errs, errs_int = [], []
        for (t, x, y, idx) in self.samples:
            if lo_idx <= idx <= hi_idx:
                e, _ = path_gen.cross_track((x, y), self.pts, idx, window=300)
                errs.append(e)
                if idx <= hi_idx_int:
                    errs_int.append(e)
        n = len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / n) if n else float('nan')
        n_i = len(errs_int)
        rmse_int = (math.sqrt(sum(e * e for e in errs_int) / n_i)
                    if n_i else float('nan'))
        # Truncation detector.  A FollowPath goal can end early (see path_gen.EXIT_M);
        # if it does, this point measured a different stretch of path from its
        # neighbours and must not be compared with them.  Record it rather than
        # silently averaging whatever samples happened to land in the window.
        expected_s = self.total_len / max(self.vel, 1e-9)
        frac = elapsed / expected_s if expected_s > 0 else 0.0
        truncated = frac < 0.90
        if truncated:
            self.get_logger().error(
                f'TRUNCATED: ran {elapsed:.1f}s of an expected {expected_s:.1f}s '
                f'({frac:.2f}); the goal was reached early. This point is not comparable.')

        out = {
            'path_kind': self.kind, 'laps': self.laps, 'spacing': self.spacing,
            'expected_s': expected_s, 'elapsed_fraction': frac, 'truncated': truncated,
            'lap_length_m': self.lap_len, 'total_length_m': self.total_len,
            'desired_vel': self.vel, 'tau_inject': self.tau,
            'tau_tot_pred': self.tau + 0.025 + 0.005,
            # This is v_commanded * tau_tot / L for THIS point -- a label, not a
            # measurement.  Averaging it over a sweep returns the prediction by
            # construction.  The observed optimum comes only from locating the
            # RMSE(v) minimum (d40 / experiment.locate).  Named to make that clear.
            'u_commanded': self.vel * (self.tau + 0.030) / 0.6,
            'rmse_m': rmse, 'n_samples_used': n, 'n_samples_total': len(self.samples),
            'window_index': [lo_idx, hi_idx], 'elapsed_s': elapsed,
            # phase-insensitive window: whole laps only, no partial corner
            'rmse_m_intlap': rmse_int, 'n_samples_intlap': n_i,
            'window_arc_m': [lo_arc, hi_arc], 'window_laps': (hi_arc - lo_arc) / self.lap_len,
            'window_arc_intlap_m': [lo_arc, hi_arc_int], 'window_laps_intlap': n_int,
            'goal_status': int(status),
            'max_err_m': max(errs) if errs else None,
            'mean_err_m': sum(errs) / n if n else None,
            # achieved delay from the delay node, [requested, mean, sd, min, max, n]
            'tau_applied_mean': self.delay_stats[1] if self.delay_stats else None,
            'tau_applied_sd': self.delay_stats[2] if self.delay_stats else None,
            'tau_applied_n': self.delay_stats[5] if self.delay_stats else None,
            'tau_requested_by_node': self.delay_stats[0] if self.delay_stats else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_file)), exist_ok=True)
        with open(self.out_file, 'w') as f:
            json.dump(out, f, indent=2)
        self.get_logger().info(
            f'RMSE {rmse:.4f} m over {n}/{len(self.samples)} samples, '
            f'{elapsed:.0f}s, status={status} -> {self.out_file}')


def main():
    rclpy.init()
    node = RunPoint()
    rc = 1
    try:
        rc = node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
