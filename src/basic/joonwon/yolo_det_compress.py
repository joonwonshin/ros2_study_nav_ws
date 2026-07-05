#!/usr/bin/env python3
"""
TurtleBot4 OAK-D 카메라 (CompressedImage 토픽) + YOLOv8 실시간 객체 탐지 노드
ROS2 Humble / Ubuntu 22.04 / TurtleBot4 환경 기준
"""

import os
import sys
import threading
from pathlib import Path
from queue import Queue, Empty

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO

BOXES_CLASS_NAME = 'Car'  # 좌표(Detection2DArray)는 이 클래스만 발행


class YOLOImageSubscriber(Node):
    def __init__(self, model, robot_namespace='robot2'):
        super().__init__('yolo_image_subscriber')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        topic_name = f'/{robot_namespace}/oakd/rgb/image_raw/compressed'
        self.get_logger().info(f"Subscribing to: {topic_name}")
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            qos_profile_sensor_data)

        self.detection_publisher = self.create_publisher(
            CompressedImage,
            '/yolo/detection/amr/compressed',
            10)

        # bbox 좌표(픽셀) 발행. compressed(rgb)와 stereo(depth) 해상도가 동일해 리사이즈 불필요
        # self.boxes_publisher = self.create_publisher(
        #     Detection2DArray,
        #     '/yolo/detection/amr/boxes',
        #     10)

        # 중심점(픽셀)만 발행
        self.center_publisher = self.create_publisher(
            PointStamped,
            '/yolo/detection/amr/center',
            10)

        # 탐지 루프를 별도 스레드에서 실행 (메인 스레드는 spin_once 전용)
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def listener_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 큐가 가득 차 있으면 오래된 프레임을 버리고 최신 프레임으로 교체
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except Empty:
                    pass
            self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            # compressed(rgb)와 stereo(depth) 해상도가 동일하므로 리사이즈 없이 좌표를 그대로 사용
            stamp = self.get_clock().now().to_msg()
            # detections_msg = Detection2DArray()
            # detections_msg.header.stamp = stamp

            results = self.model.predict(img, stream=True, verbose=False, conf=0.5)
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = f"{self.classNames[cls]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 좌표(Detection2D)는 BOXES_CLASS_NAME 클래스만 발행 -> detection_depth.py가 이 좌표로 depth 조회
                    if str(self.classNames[cls]) == BOXES_CLASS_NAME:
                        # det = Detection2D()
                        # det.header.stamp = stamp
                        # det.bbox.center.position.x = (x1 + x2) / 2.0
                        # det.bbox.center.position.y = (y1 + y2) / 2.0
                        # det.bbox.size_x = float(x2 - x1)
                        # det.bbox.size_y = float(y2 - y1)
                        # hypothesis = ObjectHypothesisWithPose()
                        # hypothesis.hypothesis.class_id = str(self.classNames[cls])
                        # hypothesis.hypothesis.score = conf
                        # det.results.append(hypothesis)
                        # detections_msg.detections.append(det)

                        center_msg = PointStamped()
                        center_msg.header.stamp = stamp
                        center_msg.point.x = (x1 + x2) / 2.0
                        center_msg.point.y = (y1 + y2) / 2.0
                        center_msg.point.z = 0.0
                        self.center_publisher.publish(center_msg)

            try:
                out_msg = self.bridge.cv2_to_compressed_imgmsg(img, dst_format='jpg')
                # out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.stamp = stamp
                self.detection_publisher.publish(out_msg)
                # self.boxes_publisher.publish(detections_msg)
            except Exception as e:
                self.get_logger().error(f"Publish failed: {e}")

            cv2.imshow("YOLOv8 Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via 'q'")
                self.should_shutdown = True
                break


def main():
    model_path = input("Enter path to model file (.pt): ").strip()
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"Unsupported model format: {suffix}")
        sys.exit(1)

    robot_namespace = input("Enter robot namespace (e.g. robot2): ").strip() or 'robot2'

    rclpy.init()
    node = YOLOImageSubscriber(model, robot_namespace)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()