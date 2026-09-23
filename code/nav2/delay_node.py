#!/usr/bin/env python3
"""
Exact-timestamp delay buffer for the Nav2 validation experiment.

Sits between the controller and the plant:

    controller_server --cmd_vel_nav--> [this node] --cmd_vel--> loopback_simulator

WHY A TIMESTAMP QUEUE AND NOT A HOLD-AND-REPUBLISH (plan v2 §4)
  The single advantage of this experiment is that tau is an injected quantity we
  know exactly.  A node that stores the latest command and republishes it on its
  own timer destroys that: it adds its own zero-order hold (T_node/2) plus jitter
  to tau, and tau_tot becomes unknown.

  So: every incoming message is stamped with its due time (arrival + tau) and
  released at that time, in order, one output per input.  The controller's 20 Hz
  cadence is preserved exactly and only shifted.  The node adds no hold of its
  own; its only error is the release-timer granularity, which is measured.

ACHIEVED DELAY IS PUBLISHED, NOT JUST LOGGED
  The release timer quantises release times, so the applied delay exceeds the
  requested one by roughly half a timer period (+0.8..1.4 ms measured, sd 0.09 ms).
  That excess is a tau-independent constant added to tau_tot -- exactly the kind
  of first-moment term Theorem 2 says must be counted -- so it is published on
  /letg2/delay_stats and run_point records it per point.
      layout: [requested, mean, sd, min, max, n]

tau IS DYNAMIC
  ros2 param set /letg2_delay_node delay_sec 0.6
  RPP already supports dynamic desired_linear_vel, so a whole sweep can run
  against one stack.  Changing tau clears the queue and the statistics.

Parameters
  delay_sec        (double, 1.0)  tau to inject
  release_rate_hz  (double, 500)  release-timer rate; must be >> controller rate
  input_topic      (string, cmd_vel_nav)
  output_topic     (string, cmd_vel)
  report_period    (double, 10.0) seconds between delay-accuracy log lines
"""
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray


class DelayNode(Node):

    def __init__(self):
        super().__init__('letg2_delay_node')
        self.declare_parameter('delay_sec', 1.0)
        self.declare_parameter('release_rate_hz', 500.0)
        self.declare_parameter('input_topic', 'cmd_vel_nav')
        self.declare_parameter('output_topic', 'cmd_vel')
        self.declare_parameter('report_period', 10.0)

        self.tau = float(self.get_parameter('delay_sec').value)
        rate = float(self.get_parameter('release_rate_hz').value)
        in_t = self.get_parameter('input_topic').value
        out_t = self.get_parameter('output_topic').value
        self.report_period = float(self.get_parameter('report_period').value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)

        self.q = deque()                 # (due_ns, arrival_ns, Twist)
        self.applied = []                # achieved delays, seconds
        self.n_in = 0
        self.n_out = 0

        self.pub = self.create_publisher(Twist, out_t, qos)
        self.sub = self.create_subscription(Twist, in_t, self.on_cmd, qos)
        self.timer = self.create_timer(1.0 / rate, self.release)
        self.report_timer = self.create_timer(self.report_period, self.report)

        self.stats_pub = self.create_publisher(Float64MultiArray, '/letg2/delay_stats', 10)
        self.stats_timer = self.create_timer(1.0, self.publish_stats)

        self.add_on_set_parameters_callback(self.on_set_params)

        self.get_logger().info(
            f'letg2 delay node: {in_t} -> [{self.tau:.4f} s] -> {out_t} '
            f'(release {rate:.0f} Hz, timestamp queue, tau is dynamic)')

    # ------------------------------------------------------------ callbacks
    def on_cmd(self, msg):
        now = self.get_clock().now().nanoseconds
        self.q.append((now + int(self.tau * 1e9), now, msg))
        self.n_in += 1

    def release(self):
        now = self.get_clock().now().nanoseconds
        while self.q and self.q[0][0] <= now:
            due, arrival, msg = self.q.popleft()
            self.pub.publish(msg)
            self.applied.append((now - arrival) * 1e-9)
            self.n_out += 1

    def on_set_params(self, params):
        for p in params:
            if p.name == 'delay_sec':
                new = float(p.value)
                if new < 0.0:
                    return SetParametersResult(successful=False, reason='delay_sec < 0')
                self.tau = new
                self.q.clear()
                self.applied.clear()
                self.n_in = self.n_out = 0
                self.get_logger().info(f'delay_sec -> {self.tau:.4f} s (queue reset)')
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------ statistics
    def _stats(self):
        a = self.applied
        if not a:
            return None
        n = len(a)
        mean = sum(a) / n
        var = sum((x - mean) ** 2 for x in a) / n
        return n, mean, var ** 0.5, min(a), max(a)

    def publish_stats(self):
        s = self._stats()
        m = Float64MultiArray()
        if s is None:
            m.data = [float(self.tau), float('nan'), float('nan'),
                      float('nan'), float('nan'), 0.0]
        else:
            n, mean, sd, lo, hi = s
            m.data = [float(self.tau), float(mean), float(sd), float(lo), float(hi), float(n)]
        self.stats_pub.publish(m)

    def report(self):
        s = self._stats()
        if s is None:
            return
        n, mean, sd, lo, hi = s
        self.get_logger().info(
            f'delay check: requested {self.tau:.4f} s | applied mean {mean:.4f} '
            f'sd {sd*1e3:.2f} ms range [{lo:.4f}, {hi:.4f}] | n={n} '
            f'in={self.n_in} out={self.n_out} queued={len(self.q)}')

    def final_report(self):
        s = self._stats()
        if s is None:
            self.get_logger().warn('no commands were delayed')
            return
        n, mean, sd, lo, hi = s
        err = 100.0 * (mean / self.tau - 1.0) if self.tau > 0 else 0.0
        self.get_logger().info(
            f'FINAL delay: requested {self.tau:.4f} s, applied {mean:.4f} s '
            f'({err:+.2f}%), sd {sd*1e3:.2f} ms, range [{lo:.4f}, {hi:.4f}], n={n}')


def main():
    rclpy.init()
    node = DelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.final_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
