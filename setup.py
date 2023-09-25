from setuptools import setup, find_packages

setup(
    name='positionedpogos', 
    version='0.1', 
    packages=find_packages(), 
    install_requires=[
        'numpy', 
        'scipy',
        'matplotlib', 
        'pydrake', 
        'underactuated'
    ]
)