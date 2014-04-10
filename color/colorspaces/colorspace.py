#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**colorspace.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *colorspace* base class.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***    External imports.
#**********************************************************************************************************************
import numpy

#**********************************************************************************************************************
#***	Internal Imports.
#**********************************************************************************************************************
import color.exceptions
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

__all__ = ["LOGGER",
		   "Colorspace"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
class Colorspace(object):
	"""
	Defines a colorspace object.
	"""

	def __init__(self,
				name,
				primaries,
				whitepoint,
				toXYZ=None,
				fromXYZ=None,
				transferFunction=None,
				inverseTransferFunction=None):
		"""
		Initializes the class.

		:param name: Colorspace name.
		:type name: str or unicode
		:param primaries: Colorspace primaries.
		:type primaries: Matrix
		:param whitepoint: Colorspace whitepoint.
		:type whitepoint: tuple or Matrix
		:param toXYZ: Transformation matrix from colorspace to *CIE XYZ* colorspace.
		:type toXYZ: Matrix
		:param fromXYZ: Transformation matrix from *CIE XYZ* colorspace to colorspace.
		:type fromXYZ: Matrix
		:param transferFunction: Colorspace transfer function.
		:type transferFunction: object
		:param inverseTransferFunction: Colorspace inverse transfer function.
		:type inverseTransferFunction: object
		"""

		# --- Setting class attributes. ---
		self.__name = None
		self.name = name
		self.__primaries = None
		self.primaries = primaries
		self.__whitepoint = None
		self.whitepoint = whitepoint
		self.__toXYZ = None
		self.toXYZ = toXYZ
		self.__fromXYZ = None
		self.fromXYZ = fromXYZ
		self.__transferFunction = None
		self.transferFunction = transferFunction
		self.__inverseTransferFunction = None
		self.inverseTransferFunction = inverseTransferFunction

	#******************************************************************************************************************
	#***	Attributes properties.
	#******************************************************************************************************************
	@property
	def name(self):
		"""
		Property for **self.__name** attribute.

		:return: self.__name.
		:rtype: str or unicode
		"""

		return self.__name

	@name.setter
	def name(self, value):
		"""
		Setter for **self.__name** attribute.

		:param value: Attribute value.
		:type value: str or unicode
		"""

		if value is not None:
			assert type(value) in (str, unicode), "'{0}' attribute: '{1}' type is not in 'str' or 'unicode'!".format(
				"name", value)
		self.__name = value

	@name.deleter
	def name(self):
		"""
		Deleter for **self.__name** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "name"))

	@property
	def primaries(self):
		"""
		Property for **self.__primaries** attribute.

		:return: self.__primaries.
		:rtype: Matrix
		"""

		return self.__primaries

	@primaries.setter
	def primaries(self, value):
		"""
		Setter for **self.__primaries** attribute.

		:param value: Attribute value.
		:type value: Matrix
		"""

		if value is not None:
			assert type(value) is numpy.matrix, "'{0}' attribute: '{1}' type is not 'numpy.matrix'!".format("primaries",
																							value)
		self.__primaries = value

	@primaries.deleter
	def primaries(self):
		"""
		Deleter for **self.__primaries** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "primaries"))

	@property
	def whitepoint(self):
		"""
		Property for **self.__whitepoint** attribute.

		:return: self.__whitepoint.
		:rtype: Matrix
		"""

		return self.__whitepoint

	@whitepoint.setter
	def whitepoint(self, value):
		"""
		Setter for **self.__whitepoint** attribute.

		:param value: Attribute value.
		:type value: Matrix
		"""

		if value is not None:
			assert type(value) in (tuple, numpy.matrix), "'{0}' attribute: '{1}' type is not 'tuple', or 'numpy.matrix'!".format("whitepoint",
																							value)
		self.__whitepoint = value

	@whitepoint.deleter
	def whitepoint(self):
		"""
		Deleter for **self.__whitepoint** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "whitepoint"))

	@property
	def toXYZ(self):
		"""
		Property for **self.__toXYZ** attribute.

		:return: self.__toXYZ.
		:rtype: Matrix
		"""

		return self.__toXYZ

	@toXYZ.setter
	def toXYZ(self, value):
		"""
		Setter for **self.__toXYZ** attribute.

		:param value: Attribute value.
		:type value: Matrix
		"""

		if value is not None:
			assert type(value) is numpy.matrix, "'{0}' attribute: '{1}' type is not 'numpy.matrix'!".format("toXYZ",
																							value)
		self.__toXYZ = value

	@toXYZ.deleter
	def toXYZ(self):
		"""
		Deleter for **self.__toXYZ** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "toXYZ"))

	@property
	def fromXYZ(self):
		"""
		Property for **self.__fromXYZ** attribute.

		:return: self.__fromXYZ.
		:rtype: Matrix
		"""

		return self.__fromXYZ

	@fromXYZ.setter
	def fromXYZ(self, value):
		"""
		Setter for **self.__fromXYZ** attribute.

		:param value: Attribute value.
		:type value: Matrix
		"""

		if value is not None:
			assert type(value) is numpy.matrix, "'{0}' attribute: '{1}' type is not 'numpy.matrix'!".format("fromXYZ",
																							value)
		self.__fromXYZ = value

	@fromXYZ.deleter
	def fromXYZ(self):
		"""
		Deleter for **self.__fromXYZ** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "fromXYZ"))

	@property
	def transferFunction(self):
		"""
		Property for **self.__transferFunction** attribute.

		:return: self.__transferFunction.
		:rtype: object
		"""

		return self.__transferFunction

	@transferFunction.setter
	def transferFunction(self, value):
		"""
		Setter for **self.__transferFunction** attribute.

		:param value: Attribute value.
		:type value: object
		"""

		if value is not None:
			assert hasattr(value, "__call__"), "'{0}' attribute: '{1}' is not callable!".format("transferFunction", value)
		self.__transferFunction = value

	@transferFunction.deleter
	def transferFunction(self):
		"""
		Deleter for **self.__transferFunction** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "transferFunction"))

	@property
	def inverseTransferFunction(self):
		"""
		Property for **self.__inverseTransferFunction** attribute.

		:return: self.__inverseTransferFunction.
		:rtype: object
		"""

		return self.__inverseTransferFunction

	@inverseTransferFunction.setter
	def inverseTransferFunction(self, value):
		"""
		Setter for **self.__inverseTransferFunction** attribute.

		:param value: Attribute value.
		:type value: object
		"""

		if value is not None:
			assert hasattr(value, "__call__"), "'{0}' attribute: '{1}' is not callable!".format("inverseTransferFunction", value)
		self.__inverseTransferFunction = value

	@inverseTransferFunction.deleter
	def inverseTransferFunction(self):
		"""
		Deleter for **self.__inverseTransferFunction** attribute.
		"""

		raise color.exceptions.ProgrammingError(
			"{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "inverseTransferFunction"))

