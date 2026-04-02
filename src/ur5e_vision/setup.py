from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur5e_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='Camera transform and object localisation utilities for UR5e robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_localizer       = ur5e_vision.object_localizer:main',
            'llm_planner            = ur5e_vision.llm_planner:main',
            'realsense_publisher    = ur5e_vision.realsense_publisher:main',
            'data_collector         = ur5e_vision.data_collector:main',
            'inspection_dashboard   = ur5e_vision.inspection_dashboard:main',
        ],
    },
)
