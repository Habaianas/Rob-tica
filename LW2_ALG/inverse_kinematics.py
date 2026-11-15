"""Labwork 2 - algebraic inverse kinematics of the UFACTORY Lite 6, as a ROS 2 node.

THIS IS THE ONLY FILE YOU EDIT IN LABWORK 2.

Implement the following, replacing every ``raise NotImplementedError``:

    Part B - the maths (no ROS needed)
        inverse_kinematics(T_d)                    every distinct solution for a pose
        select_solution(solutions, q_current)      the solution that is executed

    Part B - the node
        IkNode                                     reads a desired pose, publishes
                                                   the selected joint angles

Run the node with ``ros2 run lite6_ik ik_node``.

``transform_from_pose``, ``JOINT_LIMITS``, the constants ``TOL_SAME`` and
``TOL_WRIST`` and ``main`` below are PROVIDED: already implemented, do not edit
them.

Keep the function and class names, arguments and return types exactly as they
are: the checker node (``ik_check``), the study program (``ik_study``), the
tests in ``test/`` and the instructor's grader all use them by name.

Conventions - those of Labwork 1:

    * Frame {0} coincides with the ROS frame ``link_base``; frame {6} with
      ``link_eef``. The desired pose T_d is the pose of {6} in {0}.
    * Joint angles in RADIANS, in the controller's convention, as published on
      /joint_states (q = 0 is the controller's zero configuration).
    * Lengths in METRES.

Allowed imports: ``numpy`` and ``math`` for the maths, ``forward_kinematics``
from your own ``lite6_kinematics.forward_kinematics`` (already imported below),
and ``rclpy``, ``sensor_msgs`` and ``geometry_msgs`` for the node (already
imported below). Nothing else - no MoveIt, no kinematics library, no URDF
parser, no tf. ``test/test_inverse_kinematics.py`` checks the imports.
"""

import math

import numpy as np

from lite6_kinematics.forward_kinematics import forward_kinematics

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:         # the maths and its unit tests also run without ROS
    rclpy = None
    Node = object

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
BASE_FRAME = 'link_base'

# Modified-DH parameters of Labwork 1 used by the closed-form solution (m, rad).
# Same values as in forward_kinematics.py: d_1, a_2, a_3, d_4, d_6 and the
# offsets theta_i = q_i + offset_i of joints 2 and 3.
_D1 = 0.2435
_A2 = 0.2002
_A3 = 0.087
_D4 = 0.22761
_D6 = 0.0625
_OFF2 = -math.pi / 2
_OFF3 = -math.pi / 2

_REACH_TOL = 1e-9   # relative tolerance of the reach test (rounding error)


# =============================================================================
# Part B - the maths
# =============================================================================
def _wrap(a: float) -> float:
    """Angle a brought into (-pi, pi]."""
    a = math.remainder(a, 2.0 * math.pi)        # in [-pi, pi]
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _same(qa, qb) -> bool:
    """True if no angle of qa and qb differs by more than TOL_SAME (mod 2 pi)."""
    return all(abs(math.remainder(a - b, 2.0 * math.pi)) <= TOL_SAME
               for a, b in zip(qa, qb))


def inverse_kinematics(T_d) -> list:
    """Every distinct joint vector that puts the end-effector at T_d.

    The solution is closed-form, from the expressions of Part A: no iterative
    or optimisation method.

    Args:
        T_d: (4, 4) desired pose of {6} in {0}; translation in metres.

    Returns:
        A list of (6,) float arrays, joint angles in radians, order
        joint1..joint6, one array per distinct solution (in any order). The
        empty list if T_d is out of the robot's reach.

    Conventions of Part A:
        * every angle is in (-pi, pi]; the joint ranges JOINT_LIMITS are NOT
          applied here: a solution outside them is still returned;
        * two solutions in which no angle differs by more than TOL_SAME
          (angles compared modulo 2 pi) are the same solution: only one of
          them is returned;
        * if the angle between the axes of joints 4 and 6 is smaller than
          TOL_WRIST, those axes are aligned and every solution returned has
          q4 = 0.
    """
    T = np.asarray(T_d, dtype=float)
    R06 = T[:3, :3]
    p = T[:3, 3]

    # Wrist centre = origin of {4} = origin of {5} (a_4 = d_5 = 0): the origin
    # of {6} is d_6 along z_6 from it.
    pw = p - _D6 * R06[:, 2]
    r = math.hypot(pw[0], pw[1])
    z = pw[2] - _D1

    # Elbow: |wrist centre - origin of {2}|^2 = u^2 + v^2 depends on q3 only.
    #   K = a3 cos(th3) + d4 sin(th3) = Rr cos(th3 - phi)
    Rr = math.hypot(_A3, _D4)
    phi = math.atan2(_D4, _A3)
    K = (r * r + z * z - _A2 ** 2 - _A3 ** 2 - _D4 ** 2) / (2.0 * _A2)
    c = K / Rr
    if abs(c) > 1.0 + _REACH_TOL:
        return []                                   # out of reach
    c = max(-1.0, min(1.0, c))
    if 1.0 - abs(c) < 1e-12:                        # stretched / folded boundary
        c = math.copysign(1.0, c)
    delta = math.acos(c)

    solutions = []
    q1_base = math.atan2(pw[1], pw[0])
    for q1, X in ((q1_base, r), (q1_base + math.pi, -r)):       # shoulder: 2
        for th3 in (phi + delta, phi - delta):                  # elbow: 2
            u = _A2 + _A3 * math.cos(th3) + _D4 * math.sin(th3)
            v = -_A3 * math.sin(th3) + _D4 * math.cos(th3)
            th2 = math.atan2(-(v * X + u * z), u * X - v * z)
            q2 = th2 - _OFF2
            q3 = th3 - _OFF3

            # Orientation of the wrist: R36 = R03^T R06.
            R03 = forward_kinematics(np.array([q1, q2, q3, 0.0, 0.0, 0.0]),
                                     all_frames=True)[3][:3, :3]
            R = R03.T @ R06
            sig = math.hypot(R[0, 2], R[2, 2])          # |sin q5|
            c5 = -R[1, 2]                               # cos q5
            ang = math.atan2(sig, c5)                   # angle between axes 4 and 6

            if ang < TOL_WRIST:                         # axes aligned: q5 = 0
                q4, q5 = 0.0, 0.0
                q6 = math.atan2(R[2, 0] - R[0, 1], R[0, 0] + R[2, 1])
                wrists = [(q4, q5, q6)]
            elif math.pi - ang < TOL_WRIST:             # axes opposed: q5 = pi
                q4, q5 = 0.0, math.pi
                q6 = math.atan2(R[2, 0] + R[0, 1], R[2, 1] - R[0, 0])
                wrists = [(q4, q5, q6)]
            else:                                       # wrist: 2 (q5 > 0 and q5 < 0)
                q5 = ang
                wrists = [
                    (math.atan2(-R[2, 2], -R[0, 2]), q5, math.atan2(R[1, 1], -R[1, 0])),
                    (math.atan2(R[2, 2], R[0, 2]), -q5, math.atan2(-R[1, 1], R[1, 0])),
                ]

            for q4, q5, q6 in wrists:
                q = np.array([_wrap(q1), _wrap(q2), _wrap(q3),
                              _wrap(q4), _wrap(q5), _wrap(q6)])
                if not any(_same(q, s) for s in solutions):
                    solutions.append(q)
    return solutions


def select_solution(solutions, q_current):
    """The solution that is executed, chosen among `solutions`.

    Args:
        solutions: list of (6,) joint vectors (rad), as inverse_kinematics()
                   returns them. Their angles may be any real numbers: this
                   function must not assume they are in (-pi, pi].
        q_current: six joint angles (rad), the current configuration of the arm.

    Returns:
        A (6,) float array, the selected joint vector, or None if no solution
        is executable (no solutions at all, or all discarded). The inputs are
        not modified.

    The rule is the one of Part A: in each solution, every angle q_i is
    replaced by the value q_i + 2 pi k (k an integer) that lies inside the
    range of joint i (JOINT_LIMITS) and is closest to q_current[i]; a solution
    for which no such value exists is discarded. Among the solutions that
    remain, the one at the smallest euclidean distance from q_current is
    selected, and the returned vector is that solution with the replaced angles.
    """
    q_cur = np.asarray(q_current, dtype=float).reshape(6)
    best, best_dist = None, float('inf')
    for sol in solutions:
        s = np.asarray(sol, dtype=float).reshape(6)
        cand = np.empty(6)
        feasible = True
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            k0 = int(round((q_cur[i] - s[i]) / (2.0 * math.pi)))
            v_best, d_best = None, float('inf')
            for k in range(k0 - 2, k0 + 3):
                v = s[i] + 2.0 * math.pi * k
                if lo <= v <= hi and abs(v - q_cur[i]) < d_best:
                    v_best, d_best = v, abs(v - q_cur[i])
            if v_best is None:
                feasible = False
                break
            cand[i] = v_best
        if not feasible:
            continue
        dist = float(np.linalg.norm(cand - q_cur))
        if dist < best_dist:
            best, best_dist = cand, dist
    return best


# =============================================================================
# Part B - the node
# =============================================================================
class IkNode(Node):
    """Solves the inverse kinematics of every desired pose it receives.

    Subscribes   target_topic         geometry_msgs/PoseStamped  (/lite6_ik/target)
                 joint_states_topic   sensor_msgs/JointState     (/joint_states)
    Publishes    solution_topic       sensor_msgs/JointState     (/lite6_ik/solution)

    The target is the pose of link_eef in link_base (header.frame_id is
    BASE_FRAME). For each target received, the node computes the solutions and
    selects one from the latest joint angles received. If there is a selected
    solution it publishes a JointState:
        name             JOINT_NAMES
        position         the selected solution, in the order of JOINT_NAMES
        header.stamp     the stamp of the TARGET it answers (copy it:
                         ik_check matches the solution to the target by it)
    Otherwise it publishes nothing and logs why:
        unreachable      the pose has no solution at all
        out_of_range     it has solutions, but all of them were discarded

    The topic names are constructor arguments so that the tests can run the
    node on private topics: use them, do not write the names again.
    """

    def __init__(self, target_topic='/lite6_ik/target', joint_states_topic='/joint_states',
                 solution_topic='/lite6_ik/solution'):
        super().__init__('ik_node')
        from rclpy.qos import qos_profile_sensor_data
        self._q_latest = None       # latest joint angles (np.ndarray, 6) or None
        self._pub = self.create_publisher(JointState, solution_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, target_topic, self.on_target, 10)

    def on_joint_state(self, msg):
        """Called for every JointState: keep the latest joint angles."""
        names = list(msg.name)
        try:
            q = [msg.position[names.index(n)] for n in JOINT_NAMES]
        except (ValueError, IndexError):
            return              # message without all six joints: ignore it
        self._q_latest = np.array(q, dtype=float)

    def on_target(self, msg):
        """Called for every desired pose: solve it and publish the solution."""
        if self._q_latest is None:
            self.get_logger().warn('No joint states received yet: target ignored.')
            return

        T_d = transform_from_pose(msg)
        solutions = inverse_kinematics(T_d)
        if not solutions:
            self.get_logger().warn('unreachable: the pose has no solution. Nothing published.')
            return
        q = select_solution(solutions, self._q_latest)
        if q is None:
            self.get_logger().warn(
                f'out_of_range: all {len(solutions)} solutions are outside the joint '
                'ranges. Nothing published.')
            return

        out = JointState()
        out.header.stamp = msg.header.stamp             # copied from the TARGET
        out.name = list(JOINT_NAMES)
        out.position = [float(x) for x in q]
        self._pub.publish(out)
        self.get_logger().info(f'IK: {len(solutions)} solutions, one selected and published.')


# =============================================================================
# PROVIDED - DO NOT EDIT. Already implemented; nothing below is yours to change.
# =============================================================================
TOL_SAME = 1e-6     # rad - two solutions closer than this in every angle are the same
TOL_WRIST = 1e-6    # rad - angle between the axes of joints 4 and 6 below which they are aligned

# Joint ranges (rad) of the xarm_description model of the Lite 6, one row per
# joint, [lower, upper].
JOINT_LIMITS = np.array([
    [-math.pi * 0.99, math.pi * 0.99],
    [-2.61799, 2.61799],
    [-0.061087, math.pi * 0.99],
    [-math.pi * 0.99, math.pi * 0.99],
    [-2.1642, 2.1642],
    [-math.pi * 0.99, math.pi * 0.99],
])


def transform_from_pose(msg) -> np.ndarray:
    """(4, 4) transform of a geometry_msgs/PoseStamped.

    The quaternion is normalised first, so a slightly off unit quaternion still
    gives a rotation matrix.
    """
    p, o = msg.pose.position, msg.pose.orientation
    n = math.sqrt(o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w)
    x, y, z, w = o.x / n, o.y / n, o.z / n, o.w / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def main(args=None):
    """Entry point of ``ros2 run lite6_ik ik_node``."""
    if rclpy is None:
        raise SystemExit('ROS 2 is not available here: run this inside the container.')
    rclpy.init(args=args)
    node = None
    status = 0
    try:
        node = IkNode()
        node.get_logger().info('ik_node running: desired poses in, joint angles out')
        rclpy.spin(node)
    except NotImplementedError as e:
        print(f'\n{e}\nImplement it in lite6_ik/inverse_kinematics.py first.')
        status = 1
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == '__main__':
    raise SystemExit(main())