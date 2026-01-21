# qp_servo_node: ROS2 node for QP-based end-effector velocity control of a mobile manipulator.
# Accepts joint states and target end-effector pose (in current EE frame), outputs velocity commands at 40 Hz.
# TODO: Consider supporting servo targets for different link combinations (e.g., multiple end-effectors or mobile base).
# Note: Simultaneous goals for multiple body parts is theoretically possible with whole-body control,
# but requires more complex QP formulation and may involve trade-offs or prioritization.
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb
import qpsolvers as qp

class QPServoNode(Node):
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    DT = 1/40.0  
    INTEGRATION_DT = 0.01  # Fixed integration timestep
    POSE_ERROR_THRESHOLD = 0.02  # Threshold for considering target reached

    def __init__(self):
        super().__init__('qp_servo_node')

        qos_profile = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.arm_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states_feedback',
            self.arm_joint_state_cb,
            qos_profile
        )
        self.target_pose_sub = self.create_subscription(  #always assume in base frame
            Pose,
            'target_ee_pose',
            self.target_pose_cb,
            qos_profile
        )
        self.arm_pos_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.current_arm_joint_position = None
        self.target_pose = None 

        self.robot = rtb.models.Piper()  # Use arm-only model
        self.q_calc = None  # Store calculated q
        self.target_reached = False

    def arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position
        # Initialize q_calc with measured q if not set
        if self.q_calc is None and self.current_arm_joint_position is not None:
            self.q_calc = np.array(self.current_arm_joint_position[:6])

    def target_pose_cb(self, msg):
        position = [msg.position.x, msg.position.y, msg.position.z]
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.target_pose = SE3.Rt(UnitQuaternion(orientation).SO3(), position).A

    def control_loop(self):
        if self.target_pose is None:
              print("No target pose received yet")
              return
        # Check if we have all necessary state
        if self.current_arm_joint_position is None:
            print("No joint state received yet")
            return
        
        # Use calculated q if available, else initialize with measured q
        if self.q_calc is None:
            q = np.array(self.current_arm_joint_position[:6])
            self.q_calc = q.copy()
        else:
            q = self.q_calc
        self.robot.q = q
        wTe = self.robot.fkine(q)
        eTep = np.linalg.inv(wTe.A) @ self.target_pose
        et = np.sum(np.abs(eTep[:3, -1]))
        # If target reached, sync q_calc with measured q and stop sending commands
        if et < self.POSE_ERROR_THRESHOLD:
            self.q_calc = np.array(self.current_arm_joint_position[:6])
            if not self.target_reached:
                self.target_reached = True
                # Do not send new position commands; arm will hold last pose
            return
        self.target_reached = False
        Y = 0.01
        Q = np.eye(self.robot.n + 6)
        Q[:self.robot.n, :self.robot.n] *= Y
        Q[self.robot.n:, self.robot.n:] = (1.0 / et) * np.eye(6)
        v, _ = rtb.p_servo(wTe, self.target_pose, 1.5)
        v[3:] *= 0.5
        Aeq = np.c_[self.robot.jacobe(q), np.eye(6)]
        beq = v.reshape((6,))
        Ain = np.zeros((self.robot.n + 6, self.robot.n + 6))
        bin = np.zeros(self.robot.n + 6)
        ps = 0.1
        pi = 0.9
        Ain[:self.robot.n, :self.robot.n], bin[:self.robot.n] = self.robot.joint_velocity_damper(ps, pi, self.robot.n)
        c = np.concatenate((np.zeros(self.robot.n), np.zeros(6)))
        lb = -np.r_[self.robot.qdlim[:self.robot.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[:self.robot.n], 10 * np.ones(6)]
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
        if qd is None:
            self.get_logger().warn('QP solver failed, sending zero commands')
            self.cmd_arm([0]*6)
            return
        qd = qd[:self.robot.n]
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4
        # Integrate velocity to position using fixed dt
        q_new = q + qd * self.INTEGRATION_DT
        self.q_calc = q_new.copy()
        self.cmd_arm(q_new)
        #TODO stop sending commands when close enough to target
    
    # def cmd_base(self, v, w):
    #       twist = Twist()
    #       twist.linear.x = v
    #       twist.angular.z = w
    #       self.base_vel_cmd_pub.publish(twist)
    
    def cmd_arm(self, q):
        msg = JointState()
        assert len(q) == 6, "Expected 6 joint commands for the arm."
        msg.name = self.ARM_JOINT_NAMES
        msg.position = q
        msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_pos_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
