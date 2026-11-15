"""Labwork 1 - forward kinematics of the UFACTORY Lite 6, as a ROS 2 node.

THIS IS THE ONLY FILE YOU EDIT IN LABWORK 1.

Implement the following, replacing every ``raise NotImplementedError``:

    Part B - the maths (no ROS needed)
        dh_table(q)                        the modified DH table for a joint vector
        mdh_transform(a, alpha, d, theta)  one link transform, ^{i-1}_i T
        forward_kinematics(q, all_frames)  ^0_6 T, or all seven frames ^0_i T

    Part C - the node
        FkNode                             reads /joint_states, publishes the
                                           pose of the end-effector

Run the node with ``ros2 run lite6_kinematics fk_node``.

``pose_from_transform``, ``quaternion_from_matrix`` and ``main`` at the bottom
are PROVIDED: already implemented, do not edit them.

Keep the function and class names, arguments and return types exactly as they
are: the checker node (``fk_check``), the tests in ``test/`` and the
instructor's grader all use them by name.

Conventions - fixed, not yours to choose:

    * Craig's MODIFIED Denavit-Hartenberg convention, frames {0} ... {6}.
    * Frame {0} coincides with the ROS frame ``link_base``.
    * Frame {6} coincides with the ROS frame ``link_eef`` (the end-effector
      frame).
    * Joint angles in RADIANS, exactly as published on /joint_states
      (q = 0 is the controller's zero configuration).
    * Lengths in METRES.

Allowed imports: ``numpy`` and ``math`` for the maths, and ``rclpy``,
``sensor_msgs`` and ``geometry_msgs`` for the node (already imported below).
Nothing else - no MoveIt, no kinematics library, no URDF parser, no tf.
``numpy`` does the linear algebra (``@`` for matrix products, ``np.cos``,
``np.eye``...); the robotics is yours.
``test/test_forward_kinematics.py`` checks the imports.
"""

import numpy as np

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

# Parameters obtained in Part A (modified DH, Craig). Row i-1 describes the
# transform from frame {i-1} to frame {i}. Lengths in metres, angles in rad.
_A = (0.0, 0.0, 0.2002, 0.087, 0.0, 0.0)                                   # a_{i-1}
_ALPHA = (0.0, -np.pi / 2, np.pi, np.pi / 2, np.pi / 2, -np.pi / 2)        # alpha_{i-1}
_D = (0.2435, 0.0, 0.0, 0.22761, 0.0, 0.0625)                              # d_i
_THETA_OFFSET = (0.0, -np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0)               # theta_i = q_i + offset_i


# =============================================================================
# Part B - the maths
# =============================================================================
def dh_table(q) -> np.ndarray:
    """Modified DH parameters of the Lite 6 for joint vector q.

    Args:
        q: six joint angles (rad), controller convention, order joint1..joint6.

    Returns:
        A (6, 4) float array. Row i-1 (i = 1..6) holds, in this order:

            [ a_{i-1} (m),  alpha_{i-1} (rad),  d_i (m),  theta_i (rad) ]

        theta_i is the DH joint angle: the joint angle plus the constant
        offset between your frames and the controller's zero,
        theta_i = q_i + theta_offset_i. So dh_table(zeros(6)) is the table
        with the offsets alone.
    """
    q = np.asarray(q, dtype=float).reshape(6)
    table = np.zeros((6, 4))
    for i in range(6):
        table[i] = [_A[i], _ALPHA[i], _D[i], q[i] + _THETA_OFFSET[i]]
    return table


def mdh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Transform from frame {i-1} to frame {i}, ^{i-1}_i T, modified DH.

    Args:
        a:     a_{i-1}     (m)
        alpha: alpha_{i-1} (rad)
        d:     d_i         (m)
        theta: theta_i     (rad) - the DH angle, offset already included

    Returns:
        (4, 4) homogeneous transform.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,        -st,        0.0,  a],
        [st * ca,   ct * ca,    -sa,  -sa * d],
        [st * sa,   ct * sa,    ca,   ca * d],
        [0.0,       0.0,        0.0,  1.0],
    ])


def forward_kinematics(q, all_frames: bool = False):
    """Pose of the end-effector, ^0_6 T, for joint vector q.

    Args:
        q: six joint angles (rad), controller convention, order joint1..joint6.
        all_frames: if True, return every frame instead of only the last one.

    Returns:
        all_frames=False: (4, 4) array ^0_6 T.
        all_frames=True:  list of seven (4, 4) arrays
                          [^0_0 T (identity), ^0_1 T, ^0_2 T, ..., ^0_6 T].

    Uses dh_table(q) and mdh_transform().
    """
    T = np.eye(4)
    frames = [T.copy()]
    for a, alpha, d, theta in dh_table(q):
        T = T @ mdh_transform(a, alpha, d, theta)
        frames.append(T.copy())
    return frames if all_frames else T


# =============================================================================
# Part C - the node
# =============================================================================
class FkNode(Node):
    """Reads the joint angles and publishes the pose of the end-effector.

    Subscribes   joint_states_topic   sensor_msgs/JointState    (/joint_states)
    Publishes    pose_topic           geometry_msgs/PoseStamped (/lite6_fk/pose)

    The published message is the pose of link_eef in link_base:
        header.frame_id  BASE_FRAME
        header.stamp     the stamp of the JointState it was computed from
                         (copy it: fk_check matches the pose to the joint
                         angles by this stamp)
        pose.position    x, y, z in metres, from ^0_6 T
        pose.orientation the rotation of ^0_6 T as a quaternion; use
                         quaternion_from_matrix(), which is provided

    The topic names are constructor arguments so that the tests can run the
    node on private topics: use them, do not write the names again.
    """

    def __init__(self, joint_states_topic='/joint_states', pose_topic='/lite6_fk/pose'):
        super().__init__('fk_node')
        from rclpy.qos import qos_profile_sensor_data
        self.pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.sub = self.create_subscription(
            JointState, joint_states_topic, self.on_joint_state,
            qos_profile_sensor_data)

    def on_joint_state(self, msg):
        """Called for every JointState: compute the pose and publish it."""
        index = {name: i for i, name in enumerate(msg.name)}
        if any(n not in index for n in JOINT_NAMES):
            return                      # message without the six arm joints
        q = [msg.position[index[n]] for n in JOINT_NAMES]
        T = forward_kinematics(q)
        quat = quaternion_from_matrix(T[:3, :3])

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = BASE_FRAME
        out.pose.position.x = float(T[0, 3])
        out.pose.position.y = float(T[1, 3])
        out.pose.position.z = float(T[2, 3])
        out.pose.orientation.x = float(quat[0])
        out.pose.orientation.y = float(quat[1])
        out.pose.orientation.z = float(quat[2])
        out.pose.orientation.w = float(quat[3])
        self.pub.publish(out)


# =============================================================================
# PROVIDED - DO NOT EDIT. Already implemented; nothing below is yours to change.
# =============================================================================
def pose_from_transform(T) -> np.ndarray:
    """End-effector pose as six numbers.

    Converts ^0_6 T into the representation used by the Lite 6 controller and
    by UFACTORY Studio. fk_check uses it to print your pose.

    Args:
        T: (4, 4) homogeneous transform.

    Returns:
        (6,) array [x, y, z, roll, pitch, yaw]: position in metres, orientation
        in radians as X-Y-Z FIXED angles (Craig, section 2.8):

            R = Rz(yaw) @ Ry(pitch) @ Rx(roll),   -pi/2 <= pitch <= pi/2

        (UFACTORY Studio shows the same three angles, in degrees.) At
        pitch = +-pi/2 exactly, roll and yaw cannot be told apart; this
        function takes the degenerate-case choice of Craig 2.8: yaw = 0.
    """
    T = np.asarray(T, dtype=float)
    r = T[:3, :3]
    pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
    cb = np.cos(pitch)
    if cb > 1e-12:
        yaw = np.arctan2(r[1, 0] / cb, r[0, 0] / cb)
        roll = np.arctan2(r[2, 1] / cb, r[2, 2] / cb)
    elif pitch > 0:            # pitch = +90 deg
        yaw, roll = 0.0, np.arctan2(r[0, 1], r[1, 1])
    else:                      # pitch = -90 deg
        yaw, roll = 0.0, -np.arctan2(r[0, 1], r[1, 1])
    return np.array([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw])


def quaternion_from_matrix(R) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix, with w >= 0.

    This is the order of geometry_msgs/Quaternion: msg.x, msg.y, msg.z, msg.w.
    """
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        w, x, y, z = s / 4.0, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, s / 4.0, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4.0, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4.0
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def main(args=None):
    """Entry point of ``ros2 run lite6_kinematics fk_node``."""
    if rclpy is None:
        raise SystemExit('ROS 2 is not available here: run this inside the container.')
    rclpy.init(args=args)
    node = None
    status = 0
    try:
        node = FkNode()
        node.get_logger().info('fk_node running: joint states in, pose out')
        rclpy.spin(node)
    except NotImplementedError as e:
        print(f'\n{e}\nImplement it in lite6_kinematics/forward_kinematics.py first.')
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