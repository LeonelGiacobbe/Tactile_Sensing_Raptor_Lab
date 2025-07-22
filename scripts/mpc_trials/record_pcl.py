import rclpy
import numpy as np
import struct
from rclpy.node import Node
import time
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import CameraInfo


class PointCloudSubscriber(Node):

    def __init__(self):
        self.intrinsics_matrix = None
        self.saved_file = False

        self.pcl_subscriber = self.create_subscription(
            pc2,
            '/camera/depth/color/points',
            self.pcl_callback,
            0
        )

        self.intrinsics_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.intrinsics_callback,
            0
        )

    def intrinsics_callback(self, msg):
        self.intrinsics_matrix = msg.k
        self.get_logger().info(f"Received intrinsics matrix: ", msg.k)

    def pcl_callback(self, msg):
        # The saved npy should have a key "xyz", containing the Nx3 pcl data
        # A key called "K", containing the camera's intrinsics matrix
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
        
        points_data = point_cloud2.read_points(msg, field_names=("x", "y", "z", color_field_name), skip_nans=True)
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
            if self.intrinsics_matrix is not None:
                pcl_dict["K"] = self.intrinsics_matrix
                if not self.saved_file:
                    np.savez("pcl_recording.npz", **pcl_dict)
                    self.saved_file = True
                    self.get_logger().info("Saved pcl recording")
                else:
                    self.get_logger().info("NPZ file has already been saved. Safe to exit program")
            else:
                self.get_logger().info("Camera matrix has not been received yet. Waiting on CameraInfo publisher...")

def main(args=None):
    rclpy.init(args=args)

    pcl_subscriber = PointCloudSubscriber()
    # Sleep a few seconds to give pcl and intrinsics publisher time to initialize
    time.sleep(3)
    rclpy.spin(pcl_subscriber)
    pcl_subscriber.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()