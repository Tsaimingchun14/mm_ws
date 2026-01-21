# qp_servo_node: ROS2 node for QP-based end-effector velocity control of a mobile manipulator.
# Accepts joint states and target end-effector pose (in current base frame), outputs velocity commands at 40 Hz.
# TODO: Consider supporting servo targets for different link combinations (e.g., multiple end-effectors or mobile base).
# Note: Simultaneous goals for multiple body parts is theoretically possible with whole-body control,
# but requires more complex QP formulation and may involve trade-offs or prioritization.
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb
import qpsolvers as qp
from time import time, sleep
import matplotlib.pyplot as plt
import threading

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
        
        self.ee_pose_sub = self.create_subscription(
            Pose,
            'end_pose',
            self.ee_pose_cb,
            qos_profile
        )
        self.arm_pos_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.current_arm_joint_position = None
        self.target_pose = None
        self.ee_pose = None 

        self.robot = rtb.models.Piper()  # Use arm-only model
        self.q_calc = None  
        self.target_reached = False
        
        self.debug = True  
        if self.debug:
            self.q_calc_hist = []
            self.q_meas_hist = []
            self.ee_calc_hist = []
            self.ee_meas_hist = []
            self.time_hist = []
            self.start_time = time()
            self._plot_thread = threading.Thread(target=self.live_plot, daemon=True)
            self._plot_thread.start()

    def arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position
        # Initialize q_calc with measured q if not set
        if self.q_calc is None and self.current_arm_joint_position is not None:
            self.q_calc = np.array(self.current_arm_joint_position[:6])

    def target_pose_cb(self, msg):

        self.target_pose = [msg.position.x, msg.position.y, msg.position.z , msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        
    def ee_pose_cb(self, msg):
        self.ee_pose = [msg.position.x, msg.position.y, msg.position.z , msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    def control_loop(self):
        
        if self.target_pose is None:
            print("No target pose received yet")
            return
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
        Tep = SE3.Rt(UnitQuaternion(self.target_pose[3:]).SO3(), self.target_pose[:3]).A
        eTep = np.linalg.inv(wTe.A) @ Tep
        et = np.sum(np.abs(eTep[:3, -1]))
        # If target reached, sync q_calc with measured q and stop sending commands
        if et < self.POSE_ERROR_THRESHOLD:
            self.q_calc = np.array(self.current_arm_joint_position[:6])
            if not self.target_reached:
                self.target_reached = True
            return
        self.target_reached = False
        Y = 0.01
        Q = np.eye(self.robot.n + 6)
        Q[:self.robot.n, :self.robot.n] *= Y
        Q[self.robot.n:, self.robot.n:] = (1.0 / et) * np.eye(6)
        v, _ = rtb.p_servo(wTe, Tep, 1.5)
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
        self.q_calc = q + qd * self.INTEGRATION_DT
        self.cmd_arm(self.q_calc.tolist())
        
        if self.debug:
            self.q_calc_hist.append(q.tolist())
            self.q_meas_hist.append(self.current_arm_joint_position[:6])
            self.ee_calc_hist.append(wTe.t.tolist() + UnitQuaternion(wTe).vec.tolist())
            self.ee_meas_hist.append(self.ee_pose)
            self.time_hist.append(time() - self.start_time)
            
    
    def cmd_arm(self, q):
        msg = JointState()
        assert len(q) == 6, "Expected 6 joint commands for the arm."
        msg.name = self.ARM_JOINT_NAMES
        msg.position = q
        msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_pos_cmd_pub.publish(msg)
    
    def live_plot(self):

        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        ax_q_calc = axs[0, 0]
        ax_q_meas = axs[0, 1]
        ax_ee_calc = axs[1, 0]
        ax_ee_meas = axs[1, 1]
        # Joint angles: 6 lines, labeled as joint0 ... joint5
        lines_q_calc = [ax_q_calc.plot([], [], label=f'joint{i}')[0] for i in range(6)]
        lines_q_meas = [ax_q_meas.plot([], [], label=f'joint{i}')[0] for i in range(6)]
        # EE pose: 7 lines, labeled as x, y, z, qw, qx, qy, qz
        ee_labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        lines_ee_calc = [ax_ee_calc.plot([], [], label=lbl)[0] for lbl in ee_labels]
        lines_ee_meas = [ax_ee_meas.plot([], [], label=lbl)[0] for lbl in ee_labels]
        ax_q_calc.set_title('q_calc (joint angles)')
        ax_q_meas.set_title('q_meas (joint angles)')
        ax_ee_calc.set_title('ee_calc (pose)')
        ax_ee_meas.set_title('ee_meas (pose)')
        for ax in axs.flat:
            ax.set_xlabel('Time (s)')
        ax_q_calc.set_ylabel('Joint Value (rad)')
        ax_q_meas.set_ylabel('Joint Value (rad)')
        ax_ee_calc.set_ylabel('Pose Component')
        ax_ee_meas.set_ylabel('Pose Component')
        ax_q_calc.legend()
        ax_q_meas.legend()
        ax_ee_calc.legend()
        ax_ee_meas.legend()
        while True:
            if len(self.q_calc_hist) == 0 or len(self.q_meas_hist) == 0 or len(self.ee_calc_hist) == 0 or len(self.ee_meas_hist) == 0:
                sleep(0.05)
                continue
            t = self.time_hist
            q_calc = np.array(self.q_calc_hist)
            q_meas = np.array(self.q_meas_hist)
            ee_calc = np.array(self.ee_calc_hist)
            ee_meas = np.array(self.ee_meas_hist)
            for i in range(6):
                lines_q_calc[i].set_data(t, q_calc[:, i])
                lines_q_meas[i].set_data(t, q_meas[:, i])
            for i in range(7):
                lines_ee_calc[i].set_data(t, ee_calc[:, i])
                lines_ee_meas[i].set_data(t, ee_meas[:, i])
            for ax in axs.flat:
                ax.relim()
                ax.autoscale_view()
            plt.pause(0.05)
            sleep(0.05)

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
