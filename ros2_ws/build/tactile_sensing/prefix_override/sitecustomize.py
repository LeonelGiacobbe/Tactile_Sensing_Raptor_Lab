import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/tactile_sensing'
