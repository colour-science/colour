#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**matrix.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package matrix helper objects.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External imports.
#**********************************************************************************************************************
import numpy

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import color.verbose

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER", "isIdentity", "linearInterpolateMatrices"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
def isIdentity(matrix, n=3):
	"""
	Returns if given matrix is an identity matrix.

	Usage::

		>>> isIdentity(numpy.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
		True
		>>> isIdentity(numpy.matrix([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
		False

	:param matrix: Matrix.
	:type matrix: Matrix (N)
	:param n: Matrix dimension.
	:type n: int
	:return: Is identity matrix.
	:rtype: bool
	"""

	return numpy.array_equal(numpy.identity(n), matrix)

def linearInterpolateMatrices(a, b, matrix1, matrix2, c):
	"""
	Interpolates linearly given matrices and given base values using given interpolation value.

	Usage::

		>>> a = 2850
		>>> b = 7500
		>>> matrix1 = numpy.matrix([0.5309, -0.0229, -0.0336, -0.6241, 1.3265, 0.3337, -0.0817, 0.1215, 0.6664]).reshape((3, 3))
		>>> matrix2 = numpy.matrix([0.4716, 0.0603, -0.083, -0.7798, 1.5474, 0.248, -0.1496, 0.1937, 0.6651]).reshape((3, 3))
		>>> c = 6500
		>>> linearInterpolateMatrices(a, b, matrix1, matrix2, c)
		matrix([[ 0.48435269,  0.04240753, -0.07237634],
			[-0.74631613,  1.49989462,  0.26643011],
			[-0.13499785,  0.17817312,  0.66537957]])

	:param a: A value.
	:type a: float
	:param b: B value.
	:type b: float
	:param matrix1: Matrix 1.
	:type matrix1: Matrix (N)
	:param matrix2: Matrix 2.
	:type matrix2: Matrix (N)
	:param c: Interpolation value.
	:type c: float
	:return: Matrix.
	:rtype: Matrix (N)
	"""

	if a == b:
		return matrix1

	shape = matrix1.shape
	length = matrix1.size
	matrix1, matrix2 = numpy.ravel(matrix1), numpy.ravel(matrix2)

	# TODO: Investigate numpy implementation issues when c < a or c > b.
	# return numpy.matrix([numpy.interp(c, (a, b), zip(matrix1, matrix2)[i]) for i in range(length)]).reshape(shape)

	return numpy.matrix([matrix1[i] + (c - a) * ((matrix2[i] - matrix1[i]) / (b - a)) for i in range(length)]).reshape(shape)