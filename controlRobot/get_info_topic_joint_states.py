import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class OneTimeJointStateListener(Node):
    def __init__(self):
        super().__init__('one_time_joint_state_listener')
        self.joint_state = None
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            1
        )

    def listener_callback(self, msg):
        self.joint_state = msg
        # self.get_logger().info("Получено сообщение /joint_states")

def get_joint_state(timeout=5.0):
    rclpy.init(args=None)
    node = OneTimeJointStateListener()
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.joint_state is not None:
            break
        
    joint_state = node.joint_state
    node.destroy_node()
    rclpy.shutdown()
    return joint_state


