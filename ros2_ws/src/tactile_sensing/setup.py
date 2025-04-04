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
    ],
    install_requires=['setuptools',
                    #   'numpy',
                    #   'opencv-python',
                    #   'scipy',
                    #   'osqp',
                    #   'gelsight'
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
        ],
    },
)
