from glob import glob

from setuptools import find_packages, setup

package_name = 'mgts_go2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MGTS team',
    maintainer_email='team@example.com',
    description='3D map building and real-time localization for Unitree Go2 bags',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localizer_node = mgts_go2.localizer_node:main',
            'build_map = mgts_go2.tools.build_map:main',
            'replay = mgts_go2.tools.replay:main',
        ],
    },
)
