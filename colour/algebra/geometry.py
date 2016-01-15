#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometry
========

Defines objects related to geometrical computations:

-   :func:`normalise_vector`
-   :func:`euclidean_distance`
-   :func:`line_segments_intersections`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['normalise_vector',
           'euclidean_distance',
           'LineSegmentsIntersections_Specification',
           'line_segments_intersections']


def normalise_vector(v):
    """
    Normalises given vector :math:`v`.

    Parameters
    ----------
    v : array_like
        Vector :math:`v` to normalise.

    Returns
    -------
    ndarray
        Normalised vector :math:`v`.

    Examples
    --------
    >>> v = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> normalise_vector(v)  # doctest: +ELLIPSIS
    array([ 0.4525410...,  0.6470802...,  0.6135908...])
    """

    return v / np.linalg.norm(v)


def euclidean_distance(v_1, v_2):
    """
    Returns the euclidean distance between :math:`v_1` and :math:`v_2` vector
    arrays.

    Parameters
    ----------
    v_1 : array_like
        :math:`v_1` vector array.
    v_2 : array_like
        :math:`v_2` vector array.
    Returns
    -------
    numeric or ndarray
        Euclidean distance.

    Examples
    --------
    >>> v_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> v_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> euclidean_distance(v_1, v_2)  # doctest: +ELLIPSIS
    451.7133019...
    """

    return np.linalg.norm(np.asarray(v_1) - np.asarray(v_2), axis=-1)


class LineSegmentsIntersections_Specification(
    namedtuple('LineSegmentsIntersections_Specification',
               ('xy', 'intersect', 'parallel', 'coincident'))):
    """
    Defines the specification for intersection of line segments :math:`l_1` and
    :math:`l_2` returned by :func:`line_segments_intersections` definition.

    Parameters
    ----------
    xy : array_like
        Array storing :math:`l_1` and :math:`l_2` line segments intersections
        coordinates. Non existing segments intersections coordinates are
        specified using `np.nan` object.
    intersect : array_like
        Array of *bool* indicating if line segments :math:`l_1` and :math:`l_2`
        intersect.
    parallel : array_like
        Array of *bool* indicating if line segments :math:`l_1` and :math:`l_2`
        are parallel.
    coincident : array_like
        Array of *bool* indicating if line segments :math:`l_1` and :math:`l_2`
        are coincident.
    """


def line_segments_intersections(l_1, l_2):
    """
    Computes :math:`l_1` line segments intersections with :math:`l_2` line
    segments.

    Parameters
    ----------
    l_1 : array_like
        :math:`l_1` line segments array, each row is a line segment such as
        (:math:`x_1`, :math:`y_1`, :math:`x_2`, :math:`y_2`) where
        (:math:`x_1`, :math:`y_1`) and (:math:`x_2`, :math:`y_2`) are
        respectively the start and end points of :math:`l_1` line segments.
    l_2 : array_like
        :math:`l_2` line segments array, each row is a line segment such as
        (:math:`x_3`, :math:`y_3`, :math:`x_4`, :math:`y_4`) where
        (:math:`x_3`, :math:`y_3`) and (:math:`x_4`, :math:`y_4`) are
        respectively the start and end points of :math:`l_2` line segments.

    Returns
    -------
    LineSegmentsIntersections_Specification
        Line segments intersections specification.

    References
    ----------
    .. [1]  Bourke, P. (n.d.). Intersection point of two line segments in 2
            dimensions. Retrieved January 15, 2016, from
            http://paulbourke.net/geometry/pointlineplane/
    .. [2]  Erdem, U. M. (n.d.). Fast Line Segment Intersection. Retrieved
            January 15, 2016, from
            http://www.mathworks.com/matlabcentral/fileexchange/\
27205-fast-line-segment-intersection

    Examples
    --------
    >>> l_1 = np.array([[[0.15416284, 0.7400497],
    ...                  [0.26331502, 0.53373939]],
    ...                 [[0.01457496, 0.91874701],
    ...                  [0.90071485, 0.03342143]]])
    >>> l_2 = np.array([[[0.95694934, 0.13720932],
    ...                  [0.28382835, 0.60608318]],
    ...                 [[0.94422514, 0.85273554],
    ...                  [0.00225923, 0.52122603]],
    ...                 [[0.55203763, 0.48537741],
    ...                  [0.76813415, 0.16071675]]])
    >>> s = line_segments_intersections(l_1, l_2)
    >>> s.xy  # doctest: +ELLIPSIS
    array([[[        nan,         nan],
            [ 0.2279184...,  0.6006430...],
            [        nan,         nan]],
    <BLANKLINE>
           [[ 0.4281451...,  0.5055568...],
            [ 0.3056055...,  0.6279838...],
            [ 0.7578749...,  0.1761301...]]])
    >>> s.intersect
    array([[False,  True, False],
           [ True,  True,  True]], dtype=bool)
    >>> s.parallel
    array([[False, False, False],
           [False, False, False]], dtype=bool)
    >>> s.coincident
    array([[False, False, False],
           [False, False, False]], dtype=bool)
    """

    l_1 = np.reshape(l_1, (-1, 4))
    l_2 = np.reshape(l_2, (-1, 4))

    r_1, c_1 = l_1.shape[0], l_1.shape[1]
    r_2, c_2 = l_2.shape[0], l_2.shape[1]

    x_1, y_1, x_2, y_2 = [np.tile(l_1[:, i, np.newaxis], (1, r_2))
                          for i in range(c_1)]

    l_2 = np.transpose(l_2)

    x_3, y_3, x_4, y_4 = [np.tile(l_2[i, :], (r_1, 1))
                          for i in range(c_2)]

    x_4_x_3 = x_4 - x_3
    y_1_y_3 = y_1 - y_3
    y_4_y_3 = y_4 - y_3
    x_1_x_3 = x_1 - x_3
    x_2_x_1 = x_2 - x_1
    y_2_y_1 = y_2 - y_1

    numerator_a = x_4_x_3 * y_1_y_3 - y_4_y_3 * x_1_x_3
    numerator_b = x_2_x_1 * y_1_y_3 - y_2_y_1 * x_1_x_3
    denominator = y_4_y_3 * x_2_x_1 - x_4_x_3 * y_2_y_1

    u_a = numerator_a / denominator
    u_b = numerator_b / denominator

    intersect = np.logical_and.reduce(
        (u_a >= 0, u_a <= 1, u_b >= 0, u_b <= 1))
    xy = tstack((x_1 + x_2_x_1 * u_a, y_1 + y_2_y_1 * u_a))
    xy[~intersect] = np.nan
    parallel = denominator == 0
    coincident = np.logical_and.reduce(
        (numerator_a == 0, numerator_b == 0, parallel))

    return LineSegmentsIntersections_Specification(
        xy, intersect, parallel, coincident)
