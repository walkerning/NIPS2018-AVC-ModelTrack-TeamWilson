# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

here = os.path.dirname(os.path.abspath((__file__)))

# meta infos
NAME = "nics_at"
DESCRIPTION = ""
VERSION = "0.1"

AUTHOR = "nics"
EMAIL = ""

# package contents
MODULES = []
PACKAGES = find_packages(exclude=["tests.*", "tests"])

# dependencies
INSTALL_REQUIRES = []
TESTS_REQUIRE = []

# entry points
ENTRY_POINTS = """"""

def read_long_description(filename):
    path = os.path.join(here, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_long_description("README.md"),
    author=AUTHOR,
    author_email=EMAIL,

    py_modules=MODULES,
    packages=PACKAGES,

    entry_points=ENTRY_POINTS,
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE
)
