import rclpy
from rclpy.node import Node
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration

class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')
        self.pub = self.create_publisher(AudioNoteVector, '/robot<n>/cmd_audio', 10)
        #robot<n> 네임 스페이스는 조에 할당된 터틀봇 번호로
        
        # 1초 후에 소리 전송
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        msg = AudioNoteVector()
        msg.append = False
        msg.notes = [
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=311, max_runtime=Duration(sec=0, nanosec=200_000_000)),
            AudioNote(frequency=466, max_runtime=Duration(sec=0, nanosec=100_000_000)),
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=500_000_000)),
            AudioNote(frequency=311, max_runtime=Duration(sec=0, nanosec=200_000_000)),
            AudioNote(frequency=466, max_runtime=Duration(sec=0, nanosec=100_000_000)),
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=700_000_000)),
            AudioNote(frequency=587, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=587, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=587, max_runtime=Duration(sec=0, nanosec=300_000_000)),
            AudioNote(frequency=622, max_runtime=Duration(sec=0, nanosec=200_000_000)),
            AudioNote(frequency=466, max_runtime=Duration(sec=0, nanosec=100_000_000)),
            AudioNote(frequency=369, max_runtime=Duration(sec=0, nanosec=500_000_000)),
            AudioNote(frequency=311, max_runtime=Duration(sec=0, nanosec=200_000_000)),
            AudioNote(frequency=466, max_runtime=Duration(sec=0, nanosec=100_000_000)),
            AudioNote(frequency=392, max_runtime=Duration(sec=0, nanosec=100_000_000)),
        ]

        self.get_logger().info('삐뽀삐뽀 소리 전송 중...')
        self.pub.publish(msg)
        self.get_logger().info('삐뽀삐뽀 소리 전송 완료...')
        self.timer.cancel()  # 타이머 종료

def main(args=None):
    rclpy.init(args=args)
    node = BeepNode()
    rclpy.spin_once(node, timeout_sec=3)  # 빠르게 돌리고 종료
    node.destroy_node()
    rclpy.shutdown()
