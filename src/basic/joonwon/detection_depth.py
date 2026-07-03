#!/usr/bin/env python3
"""
/yolo/detection/amr/boxes 토픽(탐지된 bbox 좌표)을 구독하여
탐지된 물체 위치의 depth를 계산해 토픽으로 발행하는 노드.

depth 계산 방식은 depth_checker.py / depth_checker_mouse_click.py와 동일하게
CameraInfo의 K 행렬과 depth 이미지(mm)를 사용하되,
CameraInfo의 (cx, cy) 대신 탐지된 bbox 중심 좌표를 사용한다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
# from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32
import numpy as np
from cv_bridge import CvBridge

# ================================
# 설정 상수
# ================================
# AMR_DETECTION_TOPIC = '/yolo/detection/amr/compressed'
AMR_DETECTION_TOPIC = '/yolo/detection/amr/boxes'      # 탐지 bbox 좌표 토픽
DEPTH_TOPIC = '/robot2/oakd/stereo/image_raw'          # Depth 이미지 토픽
CAMERA_INFO_TOPIC = '/robot2/oakd/stereo/camera_info'  # CameraInfo 토픽
AMR_DEPTH_TOPIC = '/yolo/depth/amr'
# ================================


class DetectionDepthNode(Node):
    def __init__(self):
        super().__init__('detection_depth_node')
        self.bridge = CvBridge()
        self.K = None
        self.depth_mm = None

        # self.amr_image = None

        self.amr_subscription = self.create_subscription(
            # CompressedImage,
            Detection2DArray,
            AMR_DETECTION_TOPIC,
            self.amr_callback,
            10)

        self.depth_subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10)

        self.amr_depth_publisher = self.create_publisher(Float32, AMR_DEPTH_TOPIC, 10)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        if self.K is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        # depth_image: uint16 or float32 in mm
        self.depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def amr_callback(self, msg):
        # self.amr_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # distance_m = self.get_center_distance()
        if not msg.detections:
            return

        # 가장 신뢰도가 높은 탐지를 "the AMR"로 간주
        best = max(
            msg.detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0)

        u = int(best.bbox.center.position.x)
        v = int(best.bbox.center.position.y)
        distance_m = self.get_bbox_distance(u, v)
        if distance_m is not None:
            self.amr_depth_publisher.publish(Float32(data=distance_m))
            self.get_logger().info(f"[amr] bbox distance = {distance_m:.2f} m (u={u}, v={v})")

    # def get_center_distance(self):
    #     """depth_checker.py와 동일하게 CameraInfo의 (cx, cy) 위치의 depth 거리(m)를 반환."""
    #     if self.K is None:
    #         self.get_logger().warn('Waiting for CameraInfo...')
    #         return None
    #     if self.depth_mm is None:
    #         self.get_logger().warn('Waiting for depth image...')
    #         return None
    #
    #     u, v = int(self.K[0, 2]), int(self.K[1, 2])
    #     height, width = self.depth_mm.shape
    #     if not (0 <= u < width and 0 <= v < height):
    #         return None
    #
    #     distance_mm = self.depth_mm[v, u]
    #     return float(distance_mm) / 1000.0  # mm -> m

    def get_bbox_distance(self, u, v):
        """탐지된 bbox 중심 좌표 (u, v) 위치의 depth 거리(m)를 반환."""
        if self.depth_mm is None:
            self.get_logger().warn('Waiting for depth image...')
            return None

        height, width = self.depth_mm.shape
        if not (0 <= u < width and 0 <= v < height):
            return None

        distance_mm = self.depth_mm[v, u]
        return float(distance_mm) / 1000.0  # mm -> m


def main():
    rclpy.init()
    node = DetectionDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
