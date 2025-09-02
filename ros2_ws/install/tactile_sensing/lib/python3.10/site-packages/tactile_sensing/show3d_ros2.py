import numpy as np
import cv2
import os
import copy
from gelsight import gsdevice
from gelsight import gs3drecon
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from cv_bridge import CvBridge
import std_msgs.msg
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt16


def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5


class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')

        # Set flags
        self.CLAHE_CLIP_LIMIT = 2.0  # For lighting normalization
        self.MEDIAN_KERNEL = 3  # For depthmap post-processing
        SAVE_VIDEO_FLAG = False
        GPU = False
        MASK_MARKERS_FLAG = False
        self.USE_ROI = False
        PUBLISH_ROS_PC = True

        # Path to 3d model
        path = '.'

        # Set the camera resolution
        mmpp = 0.0634  # mini gel 18x24mm at 240x320
        self.mpp = mmpp / 1000.

        # the device ID can change after chaning the usb ports.
        # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        self.dev = gsdevice.Camera("GelSight Mini")
        net_file_path = '/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/tactile_sensing/nnmini.pt'

        self.dev.connect()

        ''' Load neural network '''
        model_file_path = path
        net_path = os.path.join(model_file_path, net_file_path)
        print('net path = ', net_path)

        if GPU:
            gpuorcpu = "cuda"
        else:
            gpuorcpu = "cpu"

        self.nn = gs3drecon.Reconstruction3D(self.dev)
        net = self.nn.load_nn(net_path, gpuorcpu)

        if SAVE_VIDEO_FLAG:
            #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
            file_path = './3dnnlive.mov'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

        f0 = self.dev.get_raw_image()

        if self.USE_ROI:
            self.roi = cv2.selectROI(f0)
            self.roi_cropped = f0[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            cv2.imshow('ROI', self.roi_cropped)
            print('Press q in ROI image to continue')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print('roi = ', self.roi)

        print('press q on image to exit')

        ''' point array to store point cloud data points '''
        x = np.arange(self.dev.imgh) * self.mpp
        y = np.arange(self.dev.imgw) * self.mpp
        X, Y = np.meshgrid(x, y)
        self.points = np.zeros([self.dev.imgw * self.dev.imgh, 3])
        self.points[:, 0] = np.ndarray.flatten(X)
        self.points[:, 1] = np.ndarray.flatten(Y)
        Z = np.zeros((self.dev.imgh, self.dev.imgw))  # initialize points array with zero depth values
        self.points[:, 2] = np.ndarray.flatten(Z)

        # I create a publisher that publishes sensor_msgs.PointCloud2 to the
        # topic 'pcd'. The value '10' refers to the history_depth, which I
        # believe is related to the ROS1 concept of queue size.
        # Read more here:
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.pcd_publisher = self.create_publisher(PointCloud2, 'pcd', 10)
        qos_profile = QoSProfile(
            depth=5,  # Last 5 messages kept
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.depth_publisher = self.create_publisher(Image, 'gs_depth', qos_profile)
        self.contact_publisher = self.create_publisher(UInt16, "/gs_contact_area", qos_profile)
        self.bridge = CvBridge();
        timer_period = 1 / 25.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _normalize_lighting(self, img):
        """Normalize lighting using CLAHE and gamma correction."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        l_norm = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        lab_norm = cv2.merge((l_norm, a, b))
        img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
        
        # Gamma correction
        img_norm = cv2.pow(img_norm / 255., 0.8) * 255
        return img_norm.astype(np.uint8)

    def timer_callback(self):

         # Get the ROI image
        f1 = self.dev.get_image()
        f1 = self._normalize_lighting(f1)
        # print("f1: ", f1)


        # Compute the depth map (dm is a 2D numpy array)
        dm = self.nn.get_depthmap(f1, False)
        dm = np.maximum(dm, 0.0)
        dm = cv2.medianBlur(dm.astype(np.float32), self.MEDIAN_KERNEL)
        
        self._publish_depth(dm)
        self._publish_pointcloud(dm)
        self._publish_contact_area(dm)

    def _publish_depth(self, dm):
        """Publish depth map as ROS Image."""
        depth_img_msg = self.bridge.cv2_to_imgmsg(
            (dm * 1000).astype(np.uint16),  # Scale for better precision
            encoding="16UC1"
        )   
        depth_img_msg.header.stamp = self.get_clock().now().to_msg()
        depth_img_msg.header.frame_id = 'map'
        self.depth_publisher.publish(depth_img_msg)

    def _publish_pointcloud(self, dm):
        """Publish point cloud."""
        dm_ros = copy.deepcopy(dm) * (0.0634 / 1000.)
        self.points[:, 2] = np.ndarray.flatten(dm_ros)
        self.pcd = point_cloud(self.points, 'map')
        self.pcd_publisher.publish(self.pcd)

    def _publish_contact_area(self, dm):
        """Calculate and publish contact area."""
        print("dm: ", dm)
        dm = np.where(dm < 1.0, 0, dm)
        normalized = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(normalized, 210, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7,7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        msg = UInt16()
        msg.data = np.count_nonzero(binary)
        self.contact_publisher.publish(msg)

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
    """
    # In a PointCloud2 message, the point cloud is stored as an byte
    # array. In order to unpack it, we also include some parameters
    # which desribes the size of each individual point.
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()

    # The fields specify what the bytes represents. The first 4 bytes
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which
    # coordinate frame it is represented in.
    header = std_msgs.msg.Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),  # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)

    pcd_publisher = PCDPublisher()
    rclpy.spin(pcd_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcd_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
