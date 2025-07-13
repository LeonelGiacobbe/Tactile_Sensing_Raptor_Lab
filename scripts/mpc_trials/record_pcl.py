import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2


class PointCloudSubscriber(Node):

    def __init__(self):
        self.pcl_subscriber = self.create_subscription(
            PointCloud2,
            'kinova_pointcloud_topic',
            self.pcl_callback,
            0
        )

    def pcl_callback(self):
        pass



def main(args=None):
    rclpy.init(args=args)

    pcl_subscriber = PointCloudSubscriber()

    rclpy.spin_once(pcl_subscriber)
    pcl_subscriber.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()