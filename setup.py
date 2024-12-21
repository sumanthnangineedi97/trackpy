from setuptools import setup, find_packages

setup(
    name='trackpy_project',
    version='1.0.0',
    description='A project for tracking cell positions using Trackpy.',
    author='Akhila Nanginedei',
    url='https://github.com/sumanthnangineedi97/trackpy',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tqdm',
        'opencv-python',
        'scipy',
        'trackpy',
        'networkx'
    ]
)
