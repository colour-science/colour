#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**chromatic_adaptation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *chromatic adaptation* data and manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.utilities.exceptions
import color.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "XYZ_SCALING_MATRIX",
           "BRADFORD_MATRIX",
           "VON_KRIES_MATRIX",
           "CAT02_MATRIX",
           "CHROMATIC_ADAPTATION_METHODS",
           "get_chromatic_adaptation_matrix"]

LOGGER = color.utilities.verbose.install_logger()

# http://brucelindbloom.com/Eqn_ChromAdapt.html
XYZ_SCALING_MATRIX = numpy.matrix(numpy.identity(3)).reshape((3, 3))

# http://brucelindbloom.com/Eqn_ChromAdapt.html
BRADFORD_MATRIX = numpy.matrix([0.8951000, 0.2664000, -0.1614000,
                                -0.7502000, 1.7135000, 0.0367000,
                                0.0389000, -0.0685000, 1.0296000]).reshape((3, 3))

# http://brucelindbloom.com/Eqn_ChromAdapt.html
VON_KRIES_MATRIX = numpy.matrix([0.4002400, 0.7076000, -0.0808100,
                                 -0.2263000, 1.1653200, 0.0457000,
                                 0.0000000, 0.0000000, 0.9182200]).reshape((3, 3))

# http://en.wikipedia.org/wiki/CIECAM02#CAT02
CAT02_MATRIX = numpy.matrix([0.7328, 0.4296, -0.1624,
                             -0.7036, 1.6975, 0.0061,
                             0.0030, 0.0136, 0.9834]).reshape((3, 3))

CHROMATIC_ADAPTATION_METHODS = {"XYZ Scaling": XYZ_SCALING_MATRIX,
                                "Bradford": BRADFORD_MATRIX,
                                "Von Kries": VON_KRIES_MATRIX,
                                "CAT02": CAT02_MATRIX}


def get_chromatic_adaptation_matrix(XYZ1, XYZ2, method="CAT02"):
    """
    Returns the *chromatic adaptation* matrix from given source and target *CIE XYZ* matrices.

    Reference: http://brucelindbloom.com/Eqn_ChromAdapt.html

    Usage::

        >>> XYZ1 = numpy.matrix([1.09923822, 1.000, 0.35445412]).reshape((3, 1))
        >>> XYZ2 = numpy.matrix([0.96907232, 1.000, 1.121792157]).reshape((3, 1)))
        >>> get_chromatic_adaptation_matrix(XYZ1, XYZ2)
        matrix([[ 0.87145615, -0.13204674,  0.40394832],
            [-0.09638805,  1.04909781,  0.1604033 ],
            [ 0.0080207 ,  0.02826367,  3.06023196]])

    :param XYZ1: *CIE XYZ* source matrix.
    :type XYZ1: matrix (3x1)
    :param XYZ2: *CIE XYZ* target matrix.
    :type XYZ2: matrix (3x1)
    :param method: Chromatic adaptation method.
    :type method: unicode
    :return: Chromatic adaptation matrix.
    :rtype: matrix (3x3)
    """

    method_matrix = CHROMATIC_ADAPTATION_METHODS.get(method)

    if method_matrix is None:
        raise color.utilities.exceptions.ProgrammingError(
            "'{0}' chromatic adaptation method is not defined! Supported methods: '{1}'.".format(method,
                                                                                                 CHROMATIC_ADAPTATION_METHODS.keys()))

    pyb_source, pyb_target = numpy.ravel(method_matrix * XYZ1), \
                             numpy.ravel(method_matrix * XYZ2)
    crd = numpy.diagflat(numpy.matrix([[pyb_target[0] / pyb_source[0],
                                        pyb_target[1] / pyb_source[1],
                                        pyb_target[2] / pyb_source[2]]])).reshape((3, 3))
    cat = method_matrix.getI() * crd * method_matrix

    LOGGER.debug("> Chromatic adaptation matrix:\n{0}".format(repr(cat)))

    return cat
