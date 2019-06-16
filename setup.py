# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', 'r') as rf:
    README = rf.read()

with open('VERSION.txt', 'r') as vf:
    VERSION = vf.read()

setup(
    name='rinokeras',
    version=VERSION,
    description='CannyLab Algorithms Repository',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Roshan Rao, David Chan',
    author_email='roshan_rao@berkeley.edu, davidchan@berkeley.edu',
    url='https://github.com/CannyLab/rinokeras',
    license='GPL-v3',
    install_requires=[
        'numpy',
        'scipy',
        'pytest',
        'tqdm',
        'h5py',
        'toposort',
        'packaging',
        'typing',
        'deprecation',
        'coverage',
        'codecov',
        'pytest-cov',
    ],
    packages=find_packages(exclude='example')  # exclude=('tests', 'docs')
)
