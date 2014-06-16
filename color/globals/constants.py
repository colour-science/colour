#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**constants.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package default constants through the :class:`Constants` class.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Constants"]


class Constants():
    """
    Defines **Color** package default constants.
    """

    application_name = "Color"
    """
    :param application_name: Package Application name.
    :type application_name: unicode
    """
    major_version = "0"
    """
    :param major_version: Package major version.
    :type major_version: unicode
    """
    minor_version = "2"
    """
    :param minor_version: Package minor version.
    :type minor_version: unicode
    """
    change_version = "0"
    """
    :param change_version: Package change version.
    :type change_version: unicode
    """
    version = ".".join((major_version, minor_version, change_version))
    """
    :param version: Package version.
    :type version: unicode
    """

    logger = "Color_Logger"
    """
    :param logger: Package logger name.
    :type logger: unicode
    """
    verbosity_level = 3
    """
    :param verbosity_level: Default logging verbosity level.
    :type verbosity_level: int
    """
    verbosity_labels = ("Critical", "Error", "Warning", "Info", "Debug")
    """
    :param verbosity_labels: Logging verbosity labels.
    :type verbosity_labels: tuple
    """
    logging_default_formatter = "Default"
    """
    :param logging_default_formatter: Default logging formatter name.
    :type logging_default_formatter: unicode
    """
    logging_separators = "*" * 96
    """
    :param logging_separators: Logging separators.
    :type logging_separators: unicode
    """

    default_codec = "utf-8"
    """
    :param default_codec: Default encoding codec.
    :type default_codec: unicode
    """
    codec_error = "ignore"
    """
    :param codec_error: Default encoding error behavior.
    :type codec_error: unicode
    """
