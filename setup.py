from setuptools import setup, find_packages

setup(
    name='floopy',
    licence='GPL 3',
    version='0.1',  # 0.1+hg4.e6508d1e0b47.local20240728
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
        'seaborn',  # tutorial.ipynb
    ],
)
