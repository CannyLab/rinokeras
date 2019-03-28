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
    author='Roshan Rao',
    author_email='roshan_rao@berkeley.edu',
    url='https://github.com/CannyLab/rl-algs',
    license=license,
    install_requires=[
        'pytest >= 3.7.0',
        'numpy >= 1.14.1',
    ],
    packages=find_packages(exclude=['example','research'])  # exclude=('tests', 'docs')
)
