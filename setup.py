# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rl_algs',
    version='0.1.0',
    description='CannyLab Algorithms Repository',
    long_description=readme,
    author='Roshan Rao',
    author_email='roshan_rao@berkeley.edu',
    url='https://github.com/CannyLab/rl-algs',
    license=license,
    packages=find_packages()  # exclude=('tests', 'docs')
)