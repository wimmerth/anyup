#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="anyup",
    version="0.0.1",
    url="https://github.com/wimmerth/anyup",
    packages=find_packages(include=["anyup", "anyup.*"]),
    license="CC-BY-4.0",
)
