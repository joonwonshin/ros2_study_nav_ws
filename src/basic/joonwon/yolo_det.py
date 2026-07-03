#!/usr/bin/env python3
"""
TurtleBot4 OAK-D 카메라 이미지 구독 + YOLOv8 실시간 객체 탐지 노드
ROS2 Humble / Ubuntu 22.04 / TurtleBot4 환경 기준

수정 사항:
  - detection_loop()를 실제로 실행하는 별도 스레드(self.thread)를 __init__에서 생성/시작
  - 메인 스레드는 rclpy.spin_once로 ROS 콜백(이미지 수신)만 처리
  - 별도 스레드는 큐에서 이미지를 꺼내 YOLO 추론 + 화면 표시를 담당
"""

import os
import sys
import threading
from pathlib import Path
from queue import Queue, Empty

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

DEPTH_SIZE = (704, 704)  # depth 이미지(704x704)와 좌표를 맞추기 위한 리사이즈 크기


class YOLOImageSubscriber(Node):
    def __init__(self, model):
        super().__init__('yolo_image_subscriber')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        # TurtleBot4 OAK-D 카메라 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            # '/robot2/oakd/rgb/image_raw/compressed',
            '/robot2/oakd/rgb/preview/image_raw',
            self.listener_callback,
            10)

        # 탐지 결과 이미지 퍼블리셔
        self.detection_publisher = self.create_publisher(
            Image,
            '/yolo/detection/amr',
            10)

        # 탐지 루프를 별도 스레드에서 실행 (메인 스레드는 spin_once 전용)
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def listener_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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

            img = cv2.resize(img, DEPTH_SIZE)   #  depth와 동일 해상도로 맞춤
            
            results = self.model.predict(img, stream=True, verbose=False)
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

            try:
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                out_msg.header.stamp = self.get_clock().now().to_msg()
                self.detection_publisher.publish(out_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish detection image: {e}")

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

    rclpy.init()
    node = YOLOImageSubscriber(model)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()