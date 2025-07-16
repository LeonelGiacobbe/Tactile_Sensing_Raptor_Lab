import rclpy
import numpy as np
import struct
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2 as pc2


class PointCloudSubscriber(Node):

    def __init__(self):
        self.pcl_subscriber = self.create_subscription(
            pc2,
            '/camera/depth/color/points',
            self.pcl_callback,
            0
        )

    def pcl_callback(self, msg):
        # The saved npy should have a key "xyz", containing the Nx3 pcl data
        # And maybe a key called "xyz_colors", containing the colors of the points
        pcl_dict = {}
        self.get_logger().debug(f"Received PointCloud message")

        color_field_name = None
        for field in msg.fields:
            if field.name == 'rgb':
                color_field_name = 'rgb'
                break

        if color_field_name is None:
            self.get_logger().warn("Point cloud does not contain 'rgb' field.")
            return
        
        points_data = pc2.read_points(msg, field_names=("x", "y", "z", color_field_name), skip_nans=True)
        xyz_coords = []
        rgb_colors = []

        for point in points_data:
            x, y, z = point[0], point[1], point[2]
            packed_color_as_float = point[3]
            xyz_coords.append([x, y, z])

            # float -> bytes -> uint32 -> RGB
            s = struct.pack('>f', packed_color_as_float)
            integer_color = struct.unpack('>I', s)[0]
            r = (integer_color >> 16) & 0x0000ff
            g = (integer_color >> 8) & 0x0000ff
            b = integer_color & 0x0000ff

            rgb_colors.append([r, g, b])

            xyz_array = np.array(xyz_coords, dtype=np.float32)
            colors_array = np.array(rgb_colors, dtype=np.uint8)

            pcl_dict["xyz"] = xyz_array
            pcl_dict["xyz_color"] = colors_array

            np.savez("pcl_recording.npz", **pcl_dict)
        

def main(args=None):
    rclpy.init(args=args)

    pcl_subscriber = PointCloudSubscriber()

    rclpy.spin_once(pcl_subscriber)
    pcl_subscriber.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()