import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion

from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

import numpy as np
import cv2
import threading
import math


class DepthToMap(Node):
    def __init__(self):
        super().__init__('depth_to_map_node')

        self.bridge = CvBridge()
        self.K = None
        self.lock = threading.Lock()

        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'

        self.depth_image = None
        self.rgb_image = None
        self.detected_point = None      # YOLO에서 받은 픽셀 좌표 (x, y)
        self.new_detection = False      # 새 탐지가 도착했는지 여부 (goal 중복 전송 방지)
        self.shutdown_requested = False
        self.display_image = None

        # self.gui_thread_stop = threading.Event()
        # self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        # self.gui_thread.start()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator()
        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()

        initial_pose = self.navigator.getPoseStamped([0.01, 0.01], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 1)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 1)
        self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, 1)

        # YOLO 탐지 중심점 구독 (마우스 클릭 대체)
        self.create_subscription(
            PointStamped,
            '/yolo/detection/amr/center',
            self.detection_callback,
            10
        )

        self.get_logger().info("TF Tree 안정화 시작. 5초 후 변환 시작합니다.")
        self.start_timer = self.create_timer(5.0, self.start_transform)

    def start_transform(self):
        self.get_logger().info("TF Tree 안정화 완료. 변환 시작합니다.")
        self.timer = self.create_timer(0.2, self.display_images)
        self.start_timer.cancel()

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
                self.logged_intrinsics = True

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                if not self.logged_depth_shape:
                    self.get_logger().info(f"Depth image received: {depth.shape}")
                    self.logged_depth_shape = True
                with self.lock:
                    self.depth_image = depth
                    self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Depth CV bridge conversion failed: {e}")

    def rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f"RGB image decoded: {rgb.shape}")
                    self.logged_rgb_shape = True
                with self.lock:
                    self.rgb_image = rgb
        except Exception as e:
            self.get_logger().error(f"Compressed RGB decode failed: {e}")

    def detection_callback(self, msg: PointStamped):
        """ YOLO 탐지 중심점(픽셀 좌표) 수신 """
        x = int(msg.point.x)
        y = int(msg.point.y)
        with self.lock:
            self.detected_point = (x, y)
            self.new_detection = True   # 새 탐지 도착 표시
        self.get_logger().info(f"Detected AMR center pixel: ({x}, {y})")

    def display_images(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            point = self.detected_point
            is_new = self.new_detection
            frame_id = getattr(self, 'camera_frame', None)

        if rgb is not None and depth is not None and frame_id:
            try:
                # rgb_display = rgb.copy()
                # depth_display = depth.copy()
                # depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
                # depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                if point is not None:
                    x, y = point
                    if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                        z = float(depth[y, x]) / 1000.0

                        if 0.2 < z < 5.0 and is_new:
                            fx, fy = self.K[0, 0], self.K[1, 1]
                            cx, cy = self.K[0, 2], self.K[1, 2]

                            X = (x - cx) * z / fx
                            Y = (y - cy) * z / fy
                            Z = z

                            pt_camera = PointStamped()
                            pt_camera.header.stamp = Time().to_msg()
                            pt_camera.header.frame_id = frame_id
                            pt_camera.point.x = X
                            pt_camera.point.y = Y
                            pt_camera.point.z = Z

                            pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=Duration(seconds=1.0))
                            self.get_logger().info(
                                f"Map coordinate: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}, {pt_map.point.z:.2f})"
                            )

                            goal_pose = PoseStamped()
                            goal_pose.header.frame_id = 'map'
                            goal_pose.header.stamp = self.get_clock().now().to_msg()
                            goal_pose.pose.position.x = pt_map.point.x
                            goal_pose.pose.position.y = pt_map.point.y
                            goal_pose.pose.position.z = 0.0
                            yaw = 0.0
                            qz = math.sin(yaw / 2.0)
                            qw = math.cos(yaw / 2.0)
                            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

                            self.navigator.goToPose(goal_pose)
                            self.get_logger().info("Sent navigation goal to detected AMR map coordinate.")

                            with self.lock:
                                self.new_detection = False   # 같은 탐지로 재전송되지 않도록 플래그 내림

                        pass
                        # cv2.circle(rgb_display, (x, y), 4, (0, 255, 0), -1)
                        # text = f"{z:.2f} m" if 0.2 < z < 5.0 else "Invalid"
                        # cv2.putText(depth_colored, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # cv2.circle(depth_colored, (x, y), 4, (255, 255, 255), -1)

                # combined = np.hstack((rgb_display, depth_colored))
                # with self.lock:
                #     self.display_image = combined.copy()
            except Exception as e:
                self.get_logger().warn(f"TF or goal error: {e}")

    # def gui_loop(self):
    #     """ 결과 뷰어만 표시 (클릭 입력 없음) """
    #     cv2.namedWindow('RGB (left) | Depth (right)', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow('RGB (left) | Depth (right)', 1280, 480)
    #     cv2.moveWindow('RGB (left) | Depth (right)', 100, 100)
    #
    #     while not self.gui_thread_stop.is_set():
    #         with self.lock:
    #             img = self.display_image.copy() if self.display_image is not None else None
    #
    #         if img is not None:
    #             cv2.imshow('RGB (left) | Depth (right)', img)
    #             key = cv2.waitKey(1)
    #             if key == ord('q'):
    #                 self.get_logger().info("Shutdown requested by user (via GUI).")
    #                 self.navigator.dock()
    #                 self.shutdown_requested = True
    #                 self.gui_thread_stop.set()
    #                 rclpy.shutdown()
    #         else:
    #             cv2.waitKey(10)


ROBOT_NAMESPACE = 'robot2'


def main():
    rclpy.init(args=[
        '--ros-args',
        '-r', f'__ns:=/{ROBOT_NAMESPACE}',
        '-r', f'/tf:=/{ROBOT_NAMESPACE}/tf',
        '-r', f'/tf_static:=/{ROBOT_NAMESPACE}/tf_static',
    ])
    node = DepthToMap()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    # node.gui_thread_stop.set()
    # node.gui_thread.join()
    node.destroy_node()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()