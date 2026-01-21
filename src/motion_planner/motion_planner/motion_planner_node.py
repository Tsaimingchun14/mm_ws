import rclpy
from rclpy.node import Node
from motion_planner.srv import SetInstruction
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import numpy as np
from cv_bridge import CvBridge
from .perception.perception import PerceptionModel

class MotionPlannerNode(Node):
    DT = 1/10.0

    def __init__(self):
        super().__init__('motion_planner_node')
        self.ee_pose_pub = self.create_publisher(Pose, 'target_ee_pose', 10)
        self.instruction_srv = self.create_service(SetInstruction, 'set_instruction', self.instruction_srv_cb)
        self.timer = self.create_timer(self.DT, self.control_loop)
        self.current_goal = dict()
        self.current_goal["text_prompt"] = None
        self.current_goal["visual_prompt"] = {"image": None, "bbox": None}
        self.active = False

        # 使用 message_filters 同步 image_raw 與 depth_image
        self.latest_image = None
        self.latest_depth = None
        self.latest_depth_info = None
        self.image_sub = message_filters.Subscriber(self, Image, 'image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth_image')
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.depth_info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_image_cb)

        # mock camera-to-base transform (4x4 numpy array)
        self.camera_to_base = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.0],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1]
        ])
        self.bridge = CvBridge()
        self.perception_model = PerceptionModel()

    def instruction_srv_cb(self, request, response):
        self.current_goal["text_prompt"] = request.text_prompt or None
        if request.image.data and request.bbox:
            self.current_goal["visual_prompt"] = {"image": request.image, "bbox": list(request.bbox)}
        else:
            self.current_goal["visual_prompt"] = None
        self.active = any([
            self.current_goal["text_prompt"],
            self.current_goal["visual_prompt"],
        ])
        response.success = True
        response.message = "Instruction updated."
        self.get_logger().info(f"SetInstruction: text={self.current_goal['text_prompt']}, \
            visual_prompt exist={self.current_goal['visual_prompt'] is not None}")
        return response

    def control_loop(self):
        if not self.active:
            return
        
        target_pose = Pose()
        visual_prompt = self.current_goal.get("visual_prompt")
        text_prompt = self.current_goal.get("text_prompt")
        grasp_center = None
        if visual_prompt and self.latest_image:
            grasp_center = self.perception_model.inference(self.latest_image, ref_img=visual_prompt["image"], bbox=visual_prompt["bbox"])
        elif text_prompt and self.latest_image:
            grasp_center = self.perception_model.inference(self.latest_image, tp=text_prompt)
     
        if grasp_center is not None:
            grasp_center_h = np.array([*grasp_center, 1.0])  # homogeneous
            grasp_center_base = self.camera_to_base @ grasp_center_h
            target_pose.position.x = grasp_center_base[0]
            target_pose.position.y = grasp_center_base[1]
            target_pose.position.z = grasp_center_base[2]
            #TODO set orientation (fixed for now)
            self.ee_pose_pub.publish(target_pose)

    def synced_image_cb(self, img_msg, depth_msg, depth_info_msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(img_msg)
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg)
        self.latest_depth_info = depth_info_msg

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
