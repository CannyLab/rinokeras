# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rinokeras',
    version='1.0.0',
    description='CannyLab Algorithms Repository',
    long_description=readme,
    author='Roshan Rao, David Chan',
    author_email='roshan_rao@berkeley.edu, davidchan@berkeley.edu',
    url='https://github.com/CannyLab/rionkeras',
    license=license,
    install_requires=[
        'numpy',
        'scipy',
        'pytest',
        'tqdm',
        'h5py',
        'packaging',
        'typing',
        'deprecation',
        'coverage',
        'codecov',
        'pytest-cov',
    ],
    packages=find_packages(exclude='example')  # exclude=('tests', 'docs')
)
