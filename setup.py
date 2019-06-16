# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

with open('VERSION.txt', 'r') as vf:
    VERSION = vf.read()

setup(
    name='rinokeras-nightly',
    version=VERSION,
    description='CannyLab Algorithms Repository',
    long_description='Read our docs: https://rinokeras.readthedocs.io/en/latest/index.html',
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
        'toposort',
        'packaging',
        'typing',
        'deprecation',
        'coverage',
        'codecov',
        'pytest-cov',
    ],
    include_package_data=True,
    setup_requires=[
          'setuptools>=41.0.1',
          'wheel>=0.33.4'],
    long_description_content_type='text/markdown',
    packages=find_packages(exclude='example')  # exclude=('tests', 'docs')
)
