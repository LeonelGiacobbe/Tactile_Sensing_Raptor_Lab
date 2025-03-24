#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')
        self.bridge = CvBridge()
        
        # Subscribe to depth topic
        self.sub = self.create_subscription(
            Image,
            '/gs_depth',
            self.depth_callback,
            10  # Queue size
        )
        self.get_logger().info("Subscribed to /gs_depth. Waiting for images...")

    def depth_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV (float32)
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # Print stats for debugging
            self.get_logger().info(
                f"Depth range: {np.min(depth_img):.2f}m to {np.max(depth_img):.2f}m"
            )
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to [0, 255] and convert to uint8
            depth_normalized = cv2.normalize(
                depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            # Apply binary threshold (adjust 127 as needed)
            _, binary_img = cv2.threshold(
                depth_normalized, 
                215,  # Threshold value
                255,  # White value
                cv2.THRESH_BINARY
            )
            white_pixels = np.count_nonzero(binary_img)
    
            # Total pixels in the image
            total_pixels = binary_img.size
            
            # Calculate percentage (scaled to [0, 1])
            self.contact_area_ = white_pixels / total_pixels
            self.get_logger().info(f"Percentage of white pixels: {self.contact_area_}")


            # Display images
            cv2.imshow("Normalized Depth", depth_normalized)
            cv2.imshow("Binary Threshold", binary_img)
            cv2.waitKey(1)  # Refresh display (1ms)
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()