#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Runs the tests suite.

**Others:**
"""

from __future__ import unicode_literals

import os
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["tests_suite"]

def _set_package_directory():
    """
    Sets the package directory in the path.

    :return: Definition success.
    :rtype: bool
    """

    package_directory = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    package_directory not in sys.path and sys.path.append(package_directory)
    return True

_set_package_directory()

def tests_suite():
    """
    Runs the tests suite.

    :return: Tests suite.
    :rtype: TestSuite
    """

    tests_loader = unittest.TestLoader()
    return tests_loader.discover(os.path.dirname(__file__))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(tests_suite())
