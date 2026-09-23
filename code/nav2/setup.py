import os
from glob import glob

from setuptools import setup

package_name = 'letg2_nav2_validation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='BongKeun Song',
    maintainer_email='bongkeun.song@fau.de',
    description='this study validation on nav2 regulated pure pursuit',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'delay_node = letg2_nav2_validation.delay_node:main',
            'run_point = letg2_nav2_validation.run_point:main',
            'loopback_midpoint = letg2_nav2_validation.loopback_midpoint:main',
            'loopback_actuator = letg2_nav2_validation.loopback_actuator:main',
            'loopback_bicycle = letg2_nav2_validation.loopback_bicycle:main',
        ],
    },
)
