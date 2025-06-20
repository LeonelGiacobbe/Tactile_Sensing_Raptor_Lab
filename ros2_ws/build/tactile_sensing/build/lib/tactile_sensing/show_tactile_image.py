import cv2
from PIL import Image
from threading import Thread, Lock
import rclpy
from rclpy.node import Node  # Enables the use of rclpy's Node class
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CameraPublisher(Node):
    """
    Create a CameraPublisher class, which is a subclass of the Node class.
    The class publishes the position of an object every 3 seconds.
    The position of the object are the x and y coordinates with respect to
    the camera frame.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """

        # Initiate the Node class's constructor and give it a name
        super().__init__('camera_publisher')

        # Create publisher(s)

        # This node publishes the position of an object every 3 seconds.
        # Maximum queue size of 10.
        self.frame_publisher_1 = self.create_publisher(Image, '/gsmini_rawimg_1', 10)
        self.frame_publisher_2 = self.create_publisher(Image, '/gsmini_rawimg_2', 10)

        # 3 seconds
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.get_image)
        self.i = 0  # Initialize a counter variable

        self.vs0 = WebcamVideoStream(src=2).start()
        self.vs1 = WebcamVideoStream(src=4).start()

        self.cvbridge = CvBridge()

    def get_image(self):
        """
        Callback function.
        This function gets called every 3 seconds.
        We locate an object using the camera and then publish its coordinates to ROS2 topics.
        """
        img_1 = self.vs0.read()
        img_2 = self.vs1.read()
        # Publish the coordinates to the topic
        self.publish_coordinates(img_1, img_2)

        # Increment counter variable
        self.i += 1

    def publish_coordinates(self, img_1, img_2):
        """
        Publish the coordinates of the object to ROS2 topics
        :param: The position of the object in centimeter coordinates [x , y]
        """
        # msg = Image()  # Create a message of this type
        msg_1 = self.cvbridge.cv2_to_imgmsg(img_1)  # Store the x and y coordinates of the object
        self.frame_publisher_1.publish(msg_1)  # Publish the position to the topic

        msg_2 = self.cvbridge.cv2_to_imgmsg(img_2)  # Store the x and y coordinates of the object
        self.frame_publisher_2.publish(msg_2)  # Publish the position to the topic



def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  +  0.5


class WebcamVideoStream :
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


def main(args=None):


    SAVE_VIDEO_FLAG = False
    SAVE_SINGLE_IMGS_FLAG = False
    cvbridge = CvBridge()
    chop_border_size = 0
    imgh = 240
    imgw = 320
    NUM_SENSORS = 1

    gs = {}
    gs['img'] = [0] * 2
    gs['gsmini_pub'] = [0] * 2
    gs['vs'] = [0] * 2
    gs['img_msg'] = [0] * 2

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    camera_publisher = CameraPublisher()

    # Spin the node so the callback function is called.
    # Publish any pending messages to the topics.
    rclpy.spin(camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()



if __name__ == '__main__':
    main()
