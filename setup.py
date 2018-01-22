from setuptools import setup


setup(
    name='lfs',
    version='0.1',
    packages=['lfs', 'lfs.elements', 'lfs.trainers'],
    install_requires=['mnist', 'numpy'],
    include_package_data=True,
)
