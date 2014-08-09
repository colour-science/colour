#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pypi Setup
==========
"""

from __future__ import unicode_literals

import codecs
import re
from setuptools import setup
from setuptools import find_packages

import colour

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["get_long_description"]


def get_long_description():
    """
    Returns the Package long description.

    Returns
    -------
    unicode
        Package long description.
    """

    description = []
    with codecs.open("README.rst", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if ".. code:: python" in line and len(description) >= 2:
                blockLine = description[-2]
                if re.search(r":$", blockLine) and not re.search(r"::$",
                                                                 blockLine):
                    description[-2] = "::".join(blockLine.rsplit(":", 1))
                continue

            description.append(line)
    return "".join(description)


setup(name="colour-science",
      version=colour.__version__,
      author=colour.__author__,
      author_email=colour.__email__,
      include_package_data=True,
      packages=find_packages(),
      scripts=[],
      url="http://github.com/colour-science/colour",
      license="",
      description="Colour is a Python colour science package implementing a \
      comprehensive number of colour theory transformations and algorithms.",
      long_description=get_long_description(),
      install_requires=["matplotlib>=1.3.1", "numpy>=1.8.1"],
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Environment :: Console",
                   "Intended Audience :: Developers",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.4",
                   "Topic :: Utilities"])
