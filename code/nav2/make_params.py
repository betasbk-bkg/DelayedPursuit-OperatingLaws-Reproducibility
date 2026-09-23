#!/usr/bin/env python3
"""
Generate a Nav2 params file for one (v, L) point of the this study sweep.

Rather than writing a params file from scratch -- which silently fails when a
required key is missing -- this patches the installed nav2_params.yaml, so every
server keeps its stock configuration and only the controller changes.

What is changed, and why (plan v2 §2.2, verified against the jazzy source):

  FollowPath -> RegulatedPurePursuitController
  desired_linear_vel                        the swept speed.  NOTE: the RPP source
                                            has no 'max_linear_vel'; the v knob is
                                            desired_linear_vel (plan v1 was wrong)
  lookahead_dist                     0.6     fixed L
  use_velocity_scaled_lookahead_dist false   else L moves with v and the sweep is void
  use_regulated_linear_velocity_scaling
                                     false   DEFAULT IS TRUE -- curvature slowdown
                                             would break the v sweep
  use_cost_regulated_linear_velocity_scaling
                                     false   DEFAULT IS TRUE
  use_rotate_to_heading              false   DEFAULT IS TRUE -- in-place rotation
                                             contaminates the trajectory
  use_collision_detection            false   triggers speed changes
  use_fixed_curvature_lookahead      false   no second carrot
  allow_reversing                    false   no cusp logic
  max_robot_pose_search_dist         2.5     pinned rather than costmap-derived, so
                                             the closest-pose window is deterministic
                                             and cannot jump a lap on a multi-lap path

  controller_frequency               20.0    fixes T_ctrl/2 = 0.025 s in tau_tot

NOT DISABLED (cannot be, see the source): approachVelocityConstraint is applied
unconditionally inside applyConstraints, using approach_velocity_scaling_dist
(default 0.6 m).  It only bites once the *transformed* plan is shorter than that,
i.e. near the true end of the path, so the RMSE window excludes the final metres
instead (plan v2 §2.3).

Usage:
  python3 make_params.py --vel 0.82 --out /tmp/p.yaml [--lookahead 0.6] [--freq 20.0]
"""
import argparse
import os

import yaml

STOCK = '/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml'


def rpp_block(vel, lookahead, search_dist):
    return {
        'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel': float(vel),
        'lookahead_dist': float(lookahead),
        'min_lookahead_dist': float(lookahead),
        'max_lookahead_dist': float(lookahead),
        'lookahead_time': 1.5,                      # unused: velocity scaling is off
        'rotate_to_heading_angular_vel': 1.8,
        'transform_tolerance': 0.1,
        'use_velocity_scaled_lookahead_dist': False,
        'min_approach_linear_velocity': 0.05,
        'approach_velocity_scaling_dist': 0.6,
        'use_collision_detection': False,
        'max_allowed_time_to_collision_up_to_carrot': 1.0,
        'use_regulated_linear_velocity_scaling': False,
        'use_cost_regulated_linear_velocity_scaling': False,
        'regulated_linear_scaling_min_radius': 0.9,
        'regulated_linear_scaling_min_speed': 0.25,
        'use_fixed_curvature_lookahead': False,
        'curvature_lookahead_dist': float(lookahead),
        'use_rotate_to_heading': False,
        'allow_reversing': False,
        'rotate_to_heading_min_angle': 0.785,
        'max_angular_accel': 3.2,
        'max_robot_pose_search_dist': float(search_dist),
        'interpolate_curvature_after_goal': False,
        'cost_scaling_dist': 0.3,
        'cost_scaling_gain': 1.0,
        'inflation_cost_scaling_factor': 3.0,
    }


def graceful_block(vel, lookahead, search_dist):
    """nav2_graceful_controller, pinned to a FIXED look-ahead.

    Defaults for k_phi / k_delta / beta / lambda are the plugin's own; they shape
    the smooth control law and are deliberately left alone -- the experiment is
    about the class, not about tuning this controller to match RPP.
    """
    return {
        'plugin': 'nav2_graceful_controller::GracefulController',
        'transform_tolerance': 0.1,
        # The target is the FARTHEST plan pose within max_lookahead, so
        # max_lookahead IS the look-ahead (up to one plan-pose spacing).
        # min_lookahead is only an instability guard and MUST stay strictly
        # below it: the selection loop (graceful_controller.cpp:219) BREAKS on
        # dist < min_lookahead, so min == max admits no candidate at all and the
        # controller then throws "Collision detected in trajectory" -- a message
        # that names the wrong cause.
        'min_lookahead': 0.05,
        'max_lookahead': float(lookahead),
        'max_robot_pose_search_dist': float(search_dist),
        'k_phi': 3.0,
        'k_delta': 2.0,
        'beta': 0.4,
        'lambda': 2.0,
        'v_linear_min': 0.05,
        'v_linear_max': float(vel),
        'v_angular_max': 5.0,
        'v_angular_min_in_place': 0.25,
        # v = min(v_max*(r/slowdown_radius), v_max/(1+beta*|k|^lambda)), then
        # clamped (smooth_control_law.cpp:66-73).  r is about the look-ahead, so
        # the shipped 1.5 would hold the robot at 40% of v_max for the whole run
        # and the swept v_linear_max would not be the speed.  Put the radius
        # below r so the proximity factor is >= 1 and drops out of the min.
        'slowdown_radius': 0.05,
        'initial_rotation': False,                  # drive the path, do not turn
        'initial_rotation_tolerance': 0.75,
        'prefer_final_rotation': False,
        'rotation_scaling_factor': 0.5,
        'allow_backward': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vel', type=float, required=True)
    ap.add_argument('--lookahead', type=float, default=0.6)
    ap.add_argument('--freq', type=float, default=20.0)
    ap.add_argument('--search-dist', type=float, default=2.5)
    ap.add_argument('--costmap-size', type=float, default=5.0,
                    help='local costmap width=height; extent = size/2 bounds the '
                         'transformed plan, so it is a VARIABLE, not a constant')
    ap.add_argument('--controller', default='rpp',
                    choices=['rpp', 'graceful'],
                    help='which controller plugin FollowPath loads; '
                         'graceful is the second implementation of the '
                         'curvature-command class')
    ap.add_argument('--stock', default=STOCK)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    with open(a.stock) as f:
        p = yaml.safe_load(f)

    cs = p['controller_server']['ros__parameters']
    cs['controller_frequency'] = float(a.freq)
    cs['FollowPath'] = (graceful_block if a.controller == 'graceful'
                       else rpp_block)(a.vel, a.lookahead, a.search_dist)
    # a slow agent must not trip the progress checker during the sweep's low-v points
    cs['progress_checker']['required_movement_radius'] = 0.1
    cs['progress_checker']['movement_time_allowance'] = 30.0
    # we drive the whole multi-lap path; do not let the goal checker cut it short
    cs['general_goal_checker']['xy_goal_tolerance'] = 0.10
    cs['general_goal_checker']['yaw_goal_tolerance'] = 6.28

    # Local costmap: strip it to nothing.
    #   - stock plugins are [voxel_layer, inflation_layer]; voxel_layer consumes /scan,
    #     which would drag in the map server and let phantom obstacles reach the
    #     controller.  With no layers the costmap is uniformly free, so costAtPose is
    #     always 0 and the only thing the costmap still does is bound the transformed
    #     plan -- which is exactly what we want it for.
    #   - stock size is 3x3 m, giving getCostmapMaxExtent() = 1.5 m.  The pre-test that
    #     produced the registered predictions used 2.5 m, so set 5x5 to match.
    #   NOTE: plugins must not be an EMPTY list -- rclcpp cannot infer the type of
    #   `plugins: []` and controller_server aborts with
    #   "parameter_value_from failed for parameter 'plugins': No parameter value set".
    #   So keep exactly one layer that needs no sensor input: inflation.  With no
    #   obstacle source it inflates nothing, so the costmap is uniformly free.
    lc = p['local_costmap']['local_costmap']['ros__parameters']
    # The graceful controller checks its simulated trajectory against the
    # costmap and there is no parameter to switch that off -- inCollision() is
    # unconditional.  With no data source the local costmap leaves every cell
    # NO_INFORMATION, which the footprint checker reads as lethal, so the
    # controller aborts on the first cycle ("Collision detected in trajectory").
    # RPP never noticed because its use_collision_detection is False.
    # Declaring the space free is the honest fix: there ARE no obstacles in this
    # experiment, and the alternative -- leaving it unknown -- asserts something
    # untrue about the world.
    lc['track_unknown_space'] = False
    lc['plugins'] = ['inflation_layer']
    lc['inflation_layer'] = {
        'plugin': 'nav2_costmap_2d::InflationLayer',
        'cost_scaling_factor': 3.0,
        'inflation_radius': 0.01,
    }
    # nav2_costmap_2d declares width/height as INTEGER metres; a float here aborts
    # controller_server with InvalidParameterTypeException (found when E2 failed).
    lc['width'] = int(round(a.costmap_size))
    lc['height'] = int(round(a.costmap_size))
    lc['rolling_window'] = True
    lc['robot_radius'] = 0.10
    lc['update_frequency'] = 20.0
    lc['publish_frequency'] = 2.0
    for k in ('voxel_layer', 'static_layer', 'obstacle_layer'):
        lc.pop(k, None)

    # Wall clock everywhere.  The loopback simulator's timer is real-time anyway, and
    # a half-configured /clock is a classic source of silent stalls in this stack.
    def force_wall_clock(node):
        if isinstance(node, dict):
            for k, val in node.items():
                if k == 'ros__parameters' and isinstance(val, dict):
                    if 'use_sim_time' in val:
                        val['use_sim_time'] = False
                force_wall_clock(val)
    force_wall_clock(p)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        yaml.safe_dump(p, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {a.out}: v={a.vel}, L={a.lookahead}, freq={a.freq}, '
          f'costmap={a.costmap_size}, search={a.search_dist}')


if __name__ == '__main__':
    main()
