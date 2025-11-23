import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import carla
import numpy as np


class CarlaRosBridgeManager:
    """Manager for CARLA ROS Bridge integration"""

    def __init__(self, carla_world, vehicles):
        self.world = carla_world
        self.vehicles = vehicles
        self.ros_node = None

    def initialize_ros(self):
        """Initialize ROS2 node and topics"""
        if not rclpy.ok():
            rclpy.init()

        self.ros_node = CarlaVehicleNode(self.vehicles)

    def publish_vehicle_states(self):
        """Publish vehicle states to ROS topics"""
        if self.ros_node:
            for i, vehicle in enumerate(self.vehicles):
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()

                # Publish pose
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                pose_msg.pose.position.x = transform.location.x
                pose_msg.pose.position.y = transform.location.y
                pose_msg.pose.position.z = transform.location.z

                self.ros_node.publish_vehicle_pose(i, pose_msg)

    def cleanup(self):
        """Cleanup ROS resources"""
        if self.ros_node:
            self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class CarlaVehicleNode(Node):
    """ROS2 node for CARLA vehicle communication"""

    def __init__(self, vehicles):
        super().__init__('carla_vehicle_node')
        self.vehicles = vehicles
        self.pose_publishers = {}

        # Create publishers for each vehicle
        for i in range(len(vehicles)):
            self.pose_publishers[i] = self.create_publisher(
                PoseStamped, f'/carla/ego_vehicle_{i}/pose', 10
            )

    def publish_vehicle_pose(self, vehicle_id, pose_msg):
        """Publish vehicle pose"""
        if vehicle_id in self.pose_publishers:
            self.pose_publishers[vehicle_id].publish(pose_msg)
