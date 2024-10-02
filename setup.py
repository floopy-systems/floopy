from setuptools import setup, find_packages
from floopy import __version__

setup(
    name='floopy',
    licence='GPL 3',
    version=__version__,
    author='Friedrich Hagedorn',
    author_email='friedrich_h@gmx.de',
    url='https://github.com/fhgd/floopy',
    description='Manage tasks with loops in a flow-based way.',
    keywords='task, test, loop, sweeps',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'collection',
        'jsonpickle',
        'tzlocal',
        'getmac',
        'rich',
    ],
)
