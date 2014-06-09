#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**lookup.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines *Nuke* *ColorLookup* node creation objects.

**Others:**

"""

# from __future__ import unicode_literals

import csv
import nuke
import os
from collections import namedtuple

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Point",
           "CurvesInformation",
           "Curve",
           "Lookup",
           "get_curve_data",
           "parse_curve_data_header",
           "parse_curve_data",
           "getCurveAxis",
           "get_curve",
           "get_lookup",
           "format_curve_data",
           "get_color_lookup_node",
           "import_curves_data_csv_file"]

COLOR_LOOKUP_CURVES_TEMPLATE = \
    """master {{{0}}}
red {{{1}}}
green {{{2}}}
blue {{{3}}}
alpha {{{4}}}"""

Point = namedtuple("Point", ("x", "y"))
CurvesInformation = namedtuple("CurvesInformation", ("curve", "axis", "values"))


class Curve(object):
    """
    Stores curve data with the :class:`Point` class.
    """

    def __init__(self, x=None, y=None):
        """
        Initializes the class.

        :param x: X axis data.
        :type x: tuple or list
        :param y: Y axis data.
        :type y: tuple or list
        """

        points = []
        if not x:
            for i in range(len(y)):
                points.append(Point(float(i) / len(y), y[i]))
        elif not y:
            for i in range(len(x)):
                points.append(Point(x[i], float(i) / len(x)))
        else:
            for i in range(len(x)):
                points.append(Point(x[i], y[i]))

        self.points = points


class Lookup(object):
    """
    Defines the lookup master, red, green, blue and alpha curves using the :class:`Curve` class.
    """

    def __init__(self, master_curve=None, red_curve=None, green_curve=None, blue_curve=None, alpha_curve=None):
        """
        Initializes the class.

        :param master_curve: Master curve.
        :type master_curve: Curve
        :param red_curve: Red curve.
        :type red_curve: Curve
        :param green_curve: Green curve.
        :type green_curve: Curve
        :param blue_curve: Blue curve.
        :type blue_curve: Curve
        :param alpha_curve: Alpha curve.
        :type alpha_curve: Curve
        """

        self.master_curve = master_curve if isinstance(master_curve, Curve) else Curve()
        self.red_curve = red_curve if isinstance(red_curve, Curve) else Curve()
        self.green_curve = green_curve if isinstance(green_curve, Curve) else Curve()
        self.blue_curve = blue_curve if isinstance(blue_curve, Curve) else Curve()
        self.alpha_curve = alpha_curve if isinstance(alpha_curve, Curve) else Curve()


def get_curve_data(file):
    """
    Reads the curve data from given CSV file.

    :param file: CSV file.
    :type file: unicode
    :return: CSV data.
    :rtype: list
    """

    with open(file, "rb") as csv_file:
        return list(csv.reader(csv_file, delimiter=","))


def parse_curve_data_header(header):
    """
    Parses the curve data header.

    :param header: Curve data header.
    :type header: list
    :return: Curves information.
    :rtype: CurvesInformation
    """

    curves_information = []
    for name, axis in map(lambda x: x.lower().split(), header):
        curves_information.append(CurvesInformation(name, axis, []))
    return curves_information


def parse_curve_data(data):
    """
    Parses the curve data.

    :param data: Curve data.
    :type data: list
    :return: Curves information.
    :rtype: CurvesInformation
    """

    curves_information = parse_curve_data_header(data.pop(0))
    for row in data:
        for i, column in enumerate(row):
            curves_information[i].values.append(column)
    return curves_information


def get_curve_axis_values(curves_information, name, axis):
    """
    Returns the curve axis values.

    :param curves_information: Curves information.
    :type curves_information: CurvesInformation
    :param name: Curve name.
    :type name: unicode
    :param axis: Axis.
    :type axis: unicode
    :return: Curves information.
    :rtype: CurvesInformation
    """

    for curve_information in curves_information:
        if curve_information.curve == name and curve_information.axis == axis:
            return curve_information.values
    return []


def get_curve(curves_information, name):
    """
    Returns a curve using given :class:`curves_information` class instance.

    :param curves_information: Curves information.
    :type curves_information: CurvesInformation
    :param name: Curve name.
    :type name: unicode
    :return: Curve.
    :rtype: Curve
    """

    return Curve(x=get_curve_axis_values(curves_information, name, "x"),
                 y=get_curve_axis_values(curves_information, name, "y"))


def get_lookup(curves_information):
    """
    Returns a :class:`Lookup` class instance using given :class:`curves_information` class instance.

    :param curves_information: Curves information.
    :type curves_information: CurvesInformation
    :return: Lookup.
    :rtype: Lookup
    """

    return Lookup(get_curve(curves_information, "master"),
                  get_curve(curves_information, "red"),
                  get_curve(curves_information, "green"),
                  get_curve(curves_information, "blue"),
                  get_curve(curves_information, "alpha"))


def format_curve_data(curve):
    """
    Formats given :class:`Curve` class instance data.

    :param curve: Curve.
    :type curve: Curve
    :return: Formatted curve data.
    :rtype: unicode
    """

    curve_data = ""
    for point in curve.points:
        curve_data += "x{0} {1} ".format(point.x, point.y)
    return "curve C {0}".format(curve_data) if curve_data is not "" else "curve C 0 1"


def get_color_lookup_node(file, template=COLOR_LOOKUP_CURVES_TEMPLATE):
    """
    Creates the *Nuke* *ColorLookup* node code using given CSV file.

    :param file: CSV file.
    :type file: unicode
    :param template: Template used for formatting.
    :type template: unicode
    :return: ColorLookup node.
    :rtype: ColorLookup
    """

    color_lookup = nuke.nodes.ColorLookup(name="ColorLookup")
    lookup = get_lookup(parse_curve_data(get_curve_data(file)))
    color_lookup.knob("lut").fromScript(template.format(format_curve_data(lookup.master_curve),
                                                        format_curve_data(lookup.red_curve),
                                                        format_curve_data(lookup.green_curve),
                                                        format_curve_data(lookup.blue_curve),
                                                        format_curve_data(lookup.alpha_curve)))
    return color_lookup


def import_curves_data_csv_file():
    """
    Import user curves data CSV file as a *Nuke* *ColorLookup* node.

    :return: ColorLookup node.
    :rtype: ColorLookup
    """

    file = nuke.getFilename("Choose ColorLookup Node Curves Data CSV File", "*.csv")
    if file is not None:
        if os.path.exists(file):
            return get_color_lookup_node(file)
