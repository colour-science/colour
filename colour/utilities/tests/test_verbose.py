# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.verbose` module.
"""

from __future__ import division, unicode_literals

import os
import sys
import unittest

from colour.utilities import (show_warning, suppress_warnings,
                              describe_environment)
from colour.utilities import warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestShowWarning', 'TestSuppressWarnings', 'TestDescribeEnvironment'
]


class TestShowWarning(unittest.TestCase):
    """
    Defines :func:`colour.utilities.verbose.show_warning` definition unit tests
    methods.
    """

    def test_show_warning(self):
        """
        Tests :func:`colour.utilities.verbose.show_warning` definition.
        """

        show_warning('This is a unit test warning!', Warning, None, None)

        with open(os.devnull) as dev_null:
            show_warning('This is a unit test warning!', Warning, None, None,
                         dev_null)

        stderr = sys.stderr
        try:
            sys.stderr = None
            show_warning('This is a unit test warning!', Warning, None, None)
        finally:
            sys.stderr = stderr


class TestSuppressWarnings(unittest.TestCase):
    """
    Defines :func:`colour.utilities.verbose.suppress_warnings` definition unit
    tests methods.
    """

    def test_suppress_warnings(self):
        """
        Tests :func:`colour.utilities.verbose.suppress_warnings` definition.
        """

        with suppress_warnings():
            warning('This is a suppressed unit test warning!')


class TestDescribeEnvironment(unittest.TestCase):
    """
    Defines :func:`colour.utilities.verbose.describe_environment` definition
    unit tests methods.
    """

    def test_describe_environment(self):
        """
        Tests :func:`colour.utilities.verbose.describe_environment` definition.
        """

        environment = describe_environment()
        self.assertIsInstance(environment, dict)
        self.assertListEqual(
            sorted(environment.keys()),
            ['Interpreter', 'Runtime', 'colour-science.org'])

        environment = describe_environment(development_packages=True)
        self.assertListEqual(
            sorted(environment.keys()),
            ['Development', 'Interpreter', 'Runtime', 'colour-science.org'])

        environment = describe_environment(
            development_packages=True, extras_packages=True)
        self.assertListEqual(
            sorted(environment.keys()), [
                'Development', 'Extras', 'Interpreter', 'Runtime',
                'colour-science.org'
            ])


if __name__ == '__main__':
    unittest.main()
