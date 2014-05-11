#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**setup.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package setup file.

**Others:**

"""

from __future__ import unicode_literals

import re
from setuptools import setup
from setuptools import find_packages

import color.globals.constants

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["get_long_description"]

def get_long_description():
    """
    Returns the Package long description.

    :return: Package long description.
    :rtype: unicode
    """

    description = []
    with open("README.rst") as file:
        for line in file:
            if ".. code:: python" in line and len(description) >= 2:
                blockLine = description[-2]
                if re.search(r":$", blockLine) and not re.search(r"::$", blockLine):
                    description[-2] = "::".join(blockLine.rsplit(":", 1))
                continue

            description.append(line)
    return "".join(description)

setup(name=color.globals.constants.Constants.application_name,
    version=color.globals.constants.Constants.version,
    author=color.globals.constants.__author__,
    author_email=color.globals.constants.__email__,
    include_package_data=True,
    packages=find_packages(),
    scripts=[],
    url="",
    license="",
    description="Color package implements color transformations objects.",
    long_description=get_long_description(),
    install_requires=["Foundations>=2.1.0", "matplotlib>=1.3.1", "numpy>=1.8.1"],
    classifiers=["Development Status :: 5 - Production/Stable",
                "Environment :: Console",
                "Intended Audience :: Developers",
                "Natural Language :: English",
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 2.7",
                "Topic :: Utilities"])
