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
            'joint_states',
            self.arm_joint_state_cb,
            qos_profile
        )
        self.target_pose_sub = self.create_subscription(  #always assume in base frame
            Pose,
            'target_ee_pose',
            self.target_pose_cb,
            qos_profile
        )
        self.base_vel_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.arm_pos_cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.current_base_joint_position = [0, 0]  # virtual joints for mobile base
        self.current_arm_joint_position = None
        self.target_pose = None 

        self.robot = rtb.models.KachakaPiper()

    def arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position

    def target_pose_cb(self, msg):
        position = [msg.position.x, msg.position.y, msg.position.z]
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.target_pose = SE3.Rt(UnitQuaternion(orientation).SO3(), position).A

    def control_loop(self):
        # Check if we have all necessary state
        if self.current_arm_joint_position is None or self.target_pose is None:
            self.cmd_base(0, 0)
            return
        q = np.zeros(self.robot.n)
        q[:2] = self.current_base_joint_position 
        q[2:8] = self.current_arm_joint_position[:6]
        self.robot.q = q
        
        wTe = self.robot.fkine(self.robot.q, include_base=False)  #  ee pose in root virtual link frame = ee pose in base frame
        eTep = np.linalg.inv(wTe) @ self.target_pose
        
        et = np.sum(np.abs(eTep[:3, -1]))
        Y = 0.01
        Q = np.eye(self.robot.n + 6)
        Q[: self.robot.n, : self.robot.n] *= Y
        Q[:2, :2] *= 1.0 / et
        Q[self.robot.n :, self.robot.n :] = (1.0 / et) * np.eye(6)
        v, _ = rtb.p_servo(wTe, self.target_pose, 1.5)
        v[3:] *= 0.5
        Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
        beq = v.reshape((6,))
        Ain = np.zeros((self.robot.n + 6, self.robot.n + 6))
        bin = np.zeros(self.robot.n + 6)
        ps = 0.1
        pi = 0.9
        Ain[: self.robot.n, : self.robot.n], bin[: self.robot.n] = self.robot.joint_velocity_damper(ps, pi, self.robot.n)
        c = np.concatenate((np.zeros(2), -self.robot.jacobm(start=self.robot.links[4]).reshape((self.robot.n - 2,)), np.zeros(6)))
        
        #Get base to face end-effector
        k_epsilon = 0.5
        bTe = self.robot.fkine(self.robot.q, include_base=False).A
        theta_epsilon = np.arctan2(bTe[1, -1], bTe[0, -1])   # oringianlly using math.atan2
        epsilon = k_epsilon * theta_epsilon
        c[0] = -epsilon
        
        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]
        
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
        if qd is None:
            self.get_logger().warn('QP solver failed, sending zero commands')
            self.cmd_base(0, 0)
            self.cmd_arm([0]*6)
            return
        qd = qd[: self.robot.n]
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4

        # Send commands to base and arm
        base_v = qd[0]
        base_w = qd[1]
        arm_qds = qd[2:8]
        self.cmd_base(base_v, base_w)
        self.cmd_arm(arm_qds)
        #TODO stop sending commands when close enough to target
        
    def cmd_base(self, v, w):
          twist = Twist()
          twist.linear.x = v
          twist.angular.z = w
          self.base_vel_cmd_pub.publish(twist)
    
    def cmd_arm(self, qds):
          msg = Float64MultiArray()
          assert len(qds) == 6, "Expected 6 joint commands for the arm."
          msg.data = [float(p) + float(q) * self.DT for p, q in zip(self.current_arm_joint_position, qds)] # integrate velocity to position TODO check joint limits
          dim = MultiArrayDimension()
          dim.label = "positions"
          dim.size = 6
          dim.stride = 6  

          msg.layout.dim = [dim]
          msg.layout.data_offset = 0
          self.arm_pos_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
