import rclpy
from rclpy.node import Node

# 소리 한 음(주파수+길이)을 표현하는 메시지 타입
from irobot_create_msgs.msg import AudioNote
# 여러 개의 AudioNote를 순서대로 담아 재생시키는 메시지 타입
from irobot_create_msgs.msg import AudioNoteVector
# 시간 길이(초/나노초)를 표현하는 표준 메시지 타입
from builtin_interfaces.msg import Duration


class BeepNode(Node):
    def __init__(self):
        # 부모 클래스(Node) 초기화. 'beep_node'는 ROS2 상에서 이 노드의 이름
        super().__init__('beep_node')

        # 'robot_name'이라는 파라미터를 선언. 실행 시 값이 안 주어지면 기본값 'robot2' 사용
        # 예: ros2 run turtlebot4_beep beep_node --ros-args -p robot_name:=robot1
        self.declare_parameter('robot_name', 'robot2')
        # 선언한 파라미터의 실제 값을 문자열로 읽어옴
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # 퍼블리셔 생성: AudioNoteVector 메시지를 '/{robot_name}/cmd_audio' 토픽으로 발행
        # 세 번째 인자 10은 QoS 큐 크기(메시지를 몇 개까지 버퍼에 쌓아둘지)
        self.publisher = self.create_publisher(
            AudioNoteVector,
            f'/{robot_name}/cmd_audio',
            10
        )

        # 1초(1.0초)마다 publish_beep 콜백을 호출하는 타이머 생성
        self.timer = self.create_timer(1.0, self.publish_beep)
        # 소리를 이미 보냈는지 여부를 기록하는 플래그 (한 번만 재생하기 위함)
        self.sent = False

    def publish_beep(self):
        # 이미 소리를 보냈다면 아무것도 하지 않고 리턴 (타이머는 계속 돌지만 재전송은 안 함)
        if self.sent:
            return

        # 발행할 메시지 객체 생성
        msg = AudioNoteVector()
        # False로 설정하면 로봇이 현재 재생 중인 소리를 덮어쓰고 이 메시지의 소리를 새로 재생
        # True였다면 현재 재생 중인 소리 뒤에 이어붙여서 재생
        msg.append = False

        # 순서대로 재생할 음의 주파수 목록 (단위: Hz). 880-440-880-440 순서로 "삐뽀삐뽀" 느낌을 냄
        frequencies = [880, 440, 880, 440]

        # 주파수 목록을 순회하며 각각을 AudioNote로 만들어 msg.notes에 추가
        for freq in frequencies:
            # 음 하나(주파수 + 재생 시간)를 담을 객체
            note = AudioNote()

            # 이 음을 얼마나 재생할지 결정하는 Duration 객체
            duration = Duration()
            duration.sec = 0            # 초 단위: 0초
            duration.nanosec = 300000000  # 나노초 단위: 3억 ns = 0.3초

            note.frequency = freq        # 현재 순회 중인 주파수 값 지정
            note.max_runtime = duration  # 위에서 만든 재생 시간(0.3초) 지정

            # 완성된 note를 메시지의 notes 배열에 추가
            msg.notes.append(note)

        # 완성된 AudioNoteVector 메시지를 토픽으로 발행 -> 로봇이 소리 재생
        self.publisher.publish(msg)
        # 터미널/로그에 전송 완료 메시지 출력
        self.get_logger().info('삐뽀 소리 전송')

        # 한 번 보냈다는 것을 표시해서, 다음 타이머 호출부터는 다시 보내지 않도록 함
        self.sent = True


def main(args=None):
    # rclpy(ROS2 파이썬 클라이언트 라이브러리) 초기화
    rclpy.init(args=args)

    # BeepNode 인스턴스 생성 (이 시점에 __init__이 실행되어 퍼블리셔/타이머가 세팅됨)
    node = BeepNode()

    try:
        # 노드를 계속 실행 상태로 유지하며 콜백(타이머 등)을 처리 (Ctrl+C 전까지 블로킹)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C로 종료 시 예외를 조용히 무시하고 아래로 진행
        pass

    # 노드 자원 정리
    node.destroy_node()
    # rclpy 종료
    rclpy.shutdown()


if __name__ == '__main__':
    main()
