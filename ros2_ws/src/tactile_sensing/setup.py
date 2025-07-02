from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tactile_sensing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'model'), glob('models/*.pth')),
    ],
    install_requires=['setuptools',
                    'torchvision',
                    'qpth',
                    'numpy',
                    'torch',
                    ],
    zip_safe=True,
    maintainer='leo',
    maintainer_email='lgiacobbe14@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'show3d_ros2 = tactile_sensing.show3d_ros2:main',
            'model_based_mpc = tactile_sensing.model_based_mpc:main',
            'nn_controller = tactile_sensing.nn_controller:main',
            'show_tac_image = tactile_sensing.show_tactile_image:main',
            'camera_publisher = tactile_sensing.camera_publisher:main',
            'multi_agent_mpc = tactile_sensing.multi_agent_mpc:main',
            'multi_agent_nn = tactile_sensing.multi_agent_nn_controller:main',
            'single_agent_nn = tactile_sensing.single_agent_nn_controller:main',
        ],
    },
)
