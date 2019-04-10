# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

setup(
    name='rinokeras',
    version='1.0.0',
    description='CannyLab Algorithms Repository',
    long_description=README,
    author='Roshan Rao, David Chan',
    author_email='roshan_rao@berkeley.edu, davidchan@berkeley.edu',
    url='https://github.com/CannyLab/rinokeras',
    license=LICENSE,
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
