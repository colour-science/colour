#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package common utilities objects that don't fall in any specific category.

**Others:**

"""

from __future__ import unicode_literals

import itertools
import os
import socket
import urllib2

import foundations.verbose
from foundations.globals.constants import Constants

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "get_steps",
           "is_uniform"]

LOGGER = foundations.verbose.install_logger()


def get_steps(distribution):
    """
    Returns the steps of given distribution.

    :param distribution: Distribution to retrieve the steps.
    :type distribution: tuple or list or Array or Matrix
    :return: steps.
    :rtype: tuple
    """

    return tuple(set([distribution[i + 1] - distribution[i] for i in range(len(distribution) - 1)]))


def is_uniform(distribution):
    """
    Returns if given distribution is uniform.

    :param distribution: Distribution to check for uniformity.
    :type distribution: tuple or list or Array or Matrix
    :return: Is uniform.
    :rtype: bool
    """

    return True if len(get_steps(distribution)) == 1 else False
