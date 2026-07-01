import rclpy
from rclpy.node import Node

from irobot_create_msgs.msg import AudioNote
from irobot_create_msgs.msg import AudioNoteVector
from builtin_interfaces.msg import Duration


class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')

        self.publisher = self.create_publisher(
            AudioNoteVector,
            '/robot2/cmd_audio',
            10
        )

        self.timer = self.create_timer(1.0, self.publish_beep)
        self.sent = False

    def publish_beep(self):
        if self.sent:
            return

        msg = AudioNoteVector()
        msg.append = False

        frequencies = [523, 659, 784, 1047]

        for freq in frequencies:
            note = AudioNote()

            duration = Duration()
            duration.sec = 0
            duration.nanosec = 300000000

            note.frequency = freq
            note.max_runtime = duration

            msg.notes.append(note)

        self.publisher.publish(msg)
        self.get_logger().info('삐뽀 소리 전송')

        self.sent = True


def main(args=None):
    rclpy.init(args=args)

    node = BeepNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()