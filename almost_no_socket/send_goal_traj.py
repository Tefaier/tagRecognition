import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

class MoveItActionClient(Node):
    def __init__(self):
        super().__init__('moveit_action_client')
        self.client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')

    def send_goal(self):
        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        point = JointTrajectoryPoint()

        pos_str = ""
        with open("data.txt", "r") as file:
            pos_str = file.readline()
        pos_str = pos_str[1 : len(pos_str)-1]
        point.positions = [float(i) for i in pos_str.split(',')] # углы суставов

        point.time_from_start.sec = 2
        trajectory.points.append(point)

        goal_msg.trajectory = trajectory
        self.client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = MoveItActionClient()
    node.send_goal()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()