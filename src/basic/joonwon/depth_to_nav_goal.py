from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion

from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

import numpy as np
import cv2
import threading
import time
import math

from message_filters import Subscriber, ApproximateTimeSynchronizer

# amr_flag가 True가 되면 이동할 좌표/방향
WEST_APPROACH_POSITION = [-2.7257, 0.3193]
WEST_APPROACH_DIRECTION = TurtleBot4Directions.WEST

# 물체와의 (depth 기준) 거리가 이 값 이하가 되면 접근을 멈추고 현재 Nav2 목표를 취소 (m)
STOP_DISTANCE = 0.5


class State(Enum):
    WAIT_FLAG = auto()   # /yolo/detection/amr_flag == True 대기
    MOVING = auto()       # 서쪽 접근 지점으로 이동 중
    DETECT = auto()       # AMR 탐지 및 접근 수행 중


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

        self.state = State.WAIT_FLAG
        self.detect_prepared = False
        self.goal_sent = False           # 접근 목표를 이미 한 번 보냈는지 여부
        self.approach_done = False      # STOP_DISTANCE 이내 도달 후 접근 종료 여부

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator()

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 1)

        # RGB/Depth를 타임스탬프 기준으로 동기화해서 함께 수신
        self.rgb_sub = Subscriber(self, CompressedImage, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        # YOLO 탐지 중심점 구독
        self.create_subscription(
            PointStamped,
            '/yolo/detection/amr/center',
            self.detection_callback,
            10
        )

        # AMR 탐지 여부 플래그 구독 (True가 되면 서쪽 접근 지점으로 이동)
        self.create_subscription(
            Bool,
            '/yolo/detection/amr_flag',
            self.amr_flag_callback,
            10
        )

        # 상태별 실행을 담당하는 메인 루프. 단일 콜백 그룹으로 묶어 중복 실행을 막는다.
        self.state_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(0.2, self.state_tick, callback_group=self.state_cb_group)

        self.get_logger().info("AMR 탐지 플래그 대기 중...")

    def amr_flag_callback(self, msg: Bool):
        """ 플래그 수신 시 상태만 변경. 실제 이동/탐지 실행은 state_tick에서 처리 """
        with self.lock:
            if self.state != State.WAIT_FLAG or not msg.data:
                return
            self.state = State.MOVING
        self.get_logger().info("AMR 탐지 플래그 수신. 이동을 시작합니다.")

    def state_tick(self):
        with self.lock:
            state = self.state

        if state == State.MOVING:
            self.handle_moving()
        elif state == State.DETECT:
            self.handle_detect()

    def handle_moving(self):
        self.get_logger().info("도킹 상태를 확인하고 초기 pose를 설정합니다.")
        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()

        initial_pose = self.navigator.getPoseStamped([0.01, 0.01], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        self.get_logger().info("서쪽 접근 지점으로 이동합니다.")
        goal_pose = self.navigator.getPoseStamped(WEST_APPROACH_POSITION, WEST_APPROACH_DIRECTION)
        self.navigator.startToPose(goal_pose)

        with self.lock:
            self.state = State.DETECT
        self.get_logger().info("접근 지점 도착.")

    def handle_detect(self):
        # 5초 TF 안정화 대기(최초 1회)만 담당. 실제 탐지 처리는 detection_callback에서
        # 새 탐지 메시지가 도착하는 즉시 실행되어 좌표 staleness를 없앤다.
        if not self.detect_prepared:
            self.detect_prepared = True
            self.get_logger().info("TF Tree 안정화 시작. 5초 후 AMR 탐지를 시작합니다.")
            time.sleep(5.0)
            self.get_logger().info("TF Tree 안정화 완료. AMR 탐지를 시작합니다.")

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
                self.logged_intrinsics = True

    def synced_callback(self, rgb_msg, depth_msg):
        """ RGB와 Depth가 타임스탬프 기준으로 동기화되어 함께 도착했을 때 호출 """
        try:
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0 and not self.logged_rgb_shape:
                self.get_logger().info(f"RGB image decoded: {rgb.shape}")
                self.logged_rgb_shape = True

            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0 and not self.logged_depth_shape:
                self.get_logger().info(f"Depth image received: {depth.shape}")
                self.logged_depth_shape = True

            with self.lock:
                if rgb is not None and rgb.size > 0:
                    self.rgb_image = rgb
                if depth is not None and depth.size > 0:
                    self.depth_image = depth
                    self.camera_frame = depth_msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Synced callback failed: {e}")

    def detection_callback(self, msg: PointStamped):
        """ YOLO 탐지 중심점(픽셀 좌표) 수신 즉시 처리 (staleness 방지) """
        x = int(msg.point.x)
        y = int(msg.point.y)
        with self.lock:
            self.detected_point = (x, y)
            ready = (
                self.state == State.DETECT
                and self.detect_prepared
                and not self.approach_done
            )
        if ready:
            self.process_detection((x, y))

    def get_patch_distance(self, depth, x, y, half_size=5):
        """ (x, y) 주변 패치의 median depth로 거리를 계산해 단일 픽셀 노이즈를 줄인다 """
        h, w = depth.shape[:2]
        y0, y1 = max(0, y - half_size), min(h, y + half_size)
        x0, x1 = max(0, x - half_size), min(w, x + half_size)
        patch = depth[y0:y1, x0:x1]
        valid = patch[patch > 0]  # 0 (측정 실패) 제외
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def process_detection(self, point):
        with self.lock:
            if self.approach_done:
                return
            depth = self.depth_image.copy() if self.depth_image is not None else None
            frame_id = getattr(self, 'camera_frame', None)
            goal_sent = self.goal_sent

        if depth is None or frame_id is None:
            return

        try:
            x, y = point
            if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
                return

            # z = float(depth[y, x]) / 1000.0
            z = self.get_patch_distance(depth, x, y)
            if z is None or not (0.2 < z < 5.0):
                return

            self.get_logger().info(f"물체와의 거리: {z:.2f} m")

            if z <= STOP_DISTANCE:
                self.navigator.cancelTask()
                with self.lock:
                    self.approach_done = True
                self.get_logger().info(f"정지 거리({STOP_DISTANCE} m) 도달. 접근을 종료합니다.")
                return

            if goal_sent:
                # 목표는 이미 한 번 보냈으므로 거리만 계속 확인하고 재전송하지 않는다
                return

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

            # 카메라(로봇) 현재 위치를 map 좌표계로 변환해 물체를 바라보는 yaw 계산
            robot_origin = PointStamped()
            robot_origin.header.stamp = Time().to_msg()
            robot_origin.header.frame_id = frame_id
            robot_origin.point.x = 0.0
            robot_origin.point.y = 0.0
            robot_origin.point.z = 0.0
            robot_pos_map = self.tf_buffer.transform(robot_origin, 'map', timeout=Duration(seconds=1.0))

            yaw = math.atan2(
                pt_map.point.y - robot_pos_map.point.y,
                pt_map.point.x - robot_pos_map.point.x
            )

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = pt_map.point.x
            goal_pose.pose.position.y = pt_map.point.y
            goal_pose.pose.position.z = 0.0
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

            self.navigator.goToPose(goal_pose)
            with self.lock:
                self.goal_sent = True
            self.get_logger().info("최초 목표를 물체 위치로 전송했습니다. 이후 거리만 계속 확인합니다.")
        except Exception as e:
            self.get_logger().warn(f"TF or goal error: {e}")


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
    node.destroy_node()


if __name__ == '__main__':
    main()
