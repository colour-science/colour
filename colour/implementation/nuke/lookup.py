#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Foundry Nuke - CSV File to ColorLookup Node
===============================================

Defines *The Foundry Nuke* *ColorLookup* node creation objects from *.csv*
files.
"""

# from __future__ import unicode_literals

import csv

try:
    import nuke
except ImportError as error:
    pass
import os
from collections import namedtuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['COLOR_LOOKUP_CURVES_TEMPLATE',
           'Point',
           'CurvesInformation',
           'Curve',
           'Lookup',
           'get_curve_data',
           'parse_curve_data_header',
           'parse_curve_data',
           'get_curve_axis_values',
           'get_curve',
           'get_lookup',
           'format_curve_data',
           'get_color_lookup_node',
           'import_curves_data_csv_file']

COLOR_LOOKUP_CURVES_TEMPLATE = (
    """master {{{0}}}
    red {{{1}}}
    green {{{2}}}
    blue {{{3}}}
    alpha {{{4}}}""")

Point = namedtuple('Point', ('x', 'y'))
CurvesInformation = namedtuple('CurvesInformation',
                               ('curve', 'axis', 'values'))


class Curve(object):
    """
    Stores curve data with the :class:`Point` class.

    Parameters
    ----------
    x : tuple or list
        x: X axis data.
    y : tuple or list
        y: Y axis data.
    """

    def __init__(self, x=None, y=None):
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
    Defines the lookup master, red, green, blue and alpha curves using the
    :class:`Curve` class.

    Parameters
    ----------
    master_curve : Curve
        master_curve: Master curve.
    red_curve : Curve
        red_curve: Red curve.
    green_curve : Curve
        green_curve: Green curve.
    blue_curve : Curve
        blue_curve: Blue curve.
    alpha_curve : Curve
        alpha_curve: Alpha curve.
    """

    def __init__(self,
                 master_curve=None,
                 red_curve=None,
                 green_curve=None,
                 blue_curve=None,
                 alpha_curve=None):
        self.master_curve = master_curve if isinstance(
            master_curve, Curve) else Curve()
        self.red_curve = red_curve if isinstance(
            red_curve, Curve) else Curve()
        self.green_curve = green_curve if isinstance(
            green_curve, Curve) else Curve()
        self.blue_curve = blue_curve if isinstance(
            blue_curve, Curve) else Curve()
        self.alpha_curve = alpha_curve if isinstance(
            alpha_curve, Curve) else Curve()


def get_curve_data(file):
    """
    Reads the curve data from given CSV file.

    Parameters
    ----------
    file : unicode
        file: CSV file.

    Returns
    -------
    list
        CSV data.
    """

    with open(file, 'rb') as csv_file:
        return tuple(csv.reader(csv_file, delimiter=','))


def parse_curve_data_header(header):
    """
    Parses the curve data header.

    Parameters
    ----------
    header : list
        header: Curve data header.

    Returns
    -------
    CurvesInformation
        Curves information.
    """

    curves_information = []
    for name, axis in [x.lower().split() for x in header]:
        curves_information.append(CurvesInformation(name, axis, []))
    return curves_information


def parse_curve_data(data):
    """
    Parses the curve data.

    Parameters
    ----------
    data : list
        data: Curve data.

    Returns
    -------
    CurvesInformation
        Curves information.
    """

    curves_information = parse_curve_data_header(data.pop(0))
    for row in data:
        for i, column in enumerate(row):
            curves_information[i].values.append(column)
    return curves_information


def get_curve_axis_values(curves_information, name, axis):
    """
    Returns the curve axis values.

    Parameters
    ----------
    curves_information : CurvesInformation
        curves_information: Curves information.
    name : unicode
        name: Curve name.
    axis : unicode
        axis: Axis.

    Returns
    -------
    CurvesInformation
        Curves information.
    """

    for curve_information in curves_information:
        if curve_information.curve == name and curve_information.axis == axis:
            return curve_information.values
    return []


def get_curve(curves_information, name):
    """
    Returns a curve using given :class:`curves_information` class instance.

    Parameters
    ----------
    curves_information : CurvesInformation
        curves_information: Curves information.
    name : unicode
        name: Curve name.

    Returns
    -------
    Curve
        Curve.
    """

    return Curve(x=get_curve_axis_values(curves_information, name, 'x'),
                 y=get_curve_axis_values(curves_information, name, 'y'))


def get_lookup(curves_information):
    """
    Returns a :class:`Lookup` class instance using given
    :class:`curves_information` class instance.

    Parameters
    ----------
    curves_information : CurvesInformation
        curves_information: Curves information.

    Returns
    -------
    Lookup
        Lookup.
    """

    return Lookup(get_curve(curves_information, 'master'),
                  get_curve(curves_information, 'red'),
                  get_curve(curves_information, 'green'),
                  get_curve(curves_information, 'blue'),
                  get_curve(curves_information, 'alpha'))


def format_curve_data(curve):
    """
    Formats given :class:`Curve` class instance data.

    Parameters
    ----------
    curve : Curve
        curve: Curve.

    Returns
    -------
    unicode
        Formatted curve data.
    """

    curve_data = ''
    for point in curve.points:
        curve_data += 'x{0} {1} '.format(point.x, point.y)
    return 'curve C {0}'.format(
        curve_data) if curve_data is not '' else 'curve C 0 1'


def get_color_lookup_node(file, template=COLOR_LOOKUP_CURVES_TEMPLATE):
    """
    Creates the *Nuke* *ColorLookup* node code using given CSV file.

    Parameters
    ----------
    file : unicode
        file: CSV file.
    template : unicode, optional
        template: Template used for formatting.

    Returns
    -------
    ColorLookup
        ColorLookup node.
    """

    color_lookup = nuke.nodes.ColorLookup(name='ColourLookup')
    lookup = get_lookup(parse_curve_data(get_curve_data(file)))
    color_lookup.knob('lut').fromScript(
        template.format(format_curve_data(lookup.master_curve),
                        format_curve_data(lookup.red_curve),
                        format_curve_data(lookup.green_curve),
                        format_curve_data(lookup.blue_curve),
                        format_curve_data(lookup.alpha_curve)))
    return color_lookup


def import_curves_data_csv_file():
    """
    Import user curves data CSV file as a *Nuke* *ColorLookup* node.

    Returns
    -------
    ColorLookup
        ColorLookup node.
    """

    file = nuke.getFilename('Choose ColorLookup Node Curves Data CSV File',
                            '*.csv')
    if file is not None:
        if os.path.exists(file):
            return get_color_lookup_node(file)
