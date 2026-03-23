from setuptools import setup

package_name = 'ur5e_scanner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot_user',
    maintainer_email='user@example.com',
    description='UR5e scanning path controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    	'console_scripts': [
        	'scanning_node = ur5e_scanner.scanning_node:main',
        	'moveit_scanning_node = ur5e_scanner.moveit_scanning_node:main',
    	],
    },
)
