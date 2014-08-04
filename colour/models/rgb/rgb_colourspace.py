# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**rgb_colourspace.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *RGB* *colourspaces* objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

import colour.algebra.common
import colour.models.cie_xyy
import colour.adaptation.cat

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RGB_Colourspace",
           "XYZ_to_RGB",
           "RGB_to_XYZ",
           "xyY_to_RGB",
           "RGB_to_xyY",
           "RGB_to_RGB"]


class RGB_Colourspace(object):
    """
    Defines a *RGB* colourspace object.
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 to_XYZ=None,
                 from_XYZ=None,
                 transfer_function=None,
                 inverse_transfer_function=None):
        """
        Initialises the class.

        :param name: *RGB* Colourspace name.
        :type name: str or unicode
        :param primaries: *RGB* Colourspace primaries.
        :type primaries: array_like
        :param whitepoint: *RGB* Colourspace whitepoint.
        :type whitepoint: array_like
        :param to_XYZ: Transformation matrix from colourspace to *CIE XYZ* colourspace.
        :type to_XYZ: array_like
        :param from_XYZ: Transformation matrix from *CIE XYZ* colourspace to colourspace.
        :type from_XYZ: array_like
        :param transfer_function: *RGB* Colourspace transfer function from linear to colourspace.
        :type transfer_function: object
        :param inverse_transfer_function: *RGB* Colourspace inverse transfer function from colourspace to linear.
        :type inverse_transfer_function: object
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__primaries = None
        self.primaries = primaries
        self.__whitepoint = None
        self.whitepoint = whitepoint
        self.__to_XYZ = None
        self.to_XYZ = to_XYZ
        self.__from_XYZ = None
        self.from_XYZ = from_XYZ
        self.__transfer_function = None
        self.transfer_function = transfer_function
        self.__inverse_transfer_function = None
        self.inverse_transfer_function = inverse_transfer_function

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

    @property
    def primaries(self):
        """
        Property for **self.__primaries** attribute.

        :return: self.__primaries.
        :rtype: array_like
        """

        return self.__primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for **self.__primaries** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = colour.algebra.common.to_ndarray(value)
        self.__primaries = value

    @property
    def whitepoint(self):
        """
        Property for **self.__whitepoint** attribute.

        :return: self.__whitepoint.
        :rtype: array_like
        """

        return self.__whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for **self.__whitepoint** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            assert type(value) in (tuple, list, np.ndarray, np.matrix), \
                "'{0}' attribute: '{1}' type is not 'tuple', 'list', 'ndarray' or 'matrix'!".format("whitepoint", value)
        self.__whitepoint = value

    @property
    def to_XYZ(self):
        """
        Property for **self.__to_XYZ** attribute.

        :return: self.__to_XYZ.
        :rtype: array_like
        """

        return self.__to_XYZ

    @to_XYZ.setter
    def to_XYZ(self, value):
        """
        Setter for **self.__to_XYZ** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = colour.algebra.common.to_ndarray(value)
        self.__to_XYZ = value

    @property
    def from_XYZ(self):
        """
        Property for **self.__from_XYZ** attribute.

        :return: self.__from_XYZ.
        :rtype: array_like
        """

        return self.__from_XYZ

    @from_XYZ.setter
    def from_XYZ(self, value):
        """
        Setter for **self.__from_XYZ** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = colour.algebra.common.to_ndarray(value)
        self.__from_XYZ = value

    @property
    def transfer_function(self):
        """
        Property for **self.__transfer_function** attribute.

        :return: self.__transfer_function.
        :rtype: object
        """

        return self.__transfer_function

    @transfer_function.setter
    def transfer_function(self, value):
        """
        Setter for **self.__transfer_function** attribute.

        :param value: Attribute value.
        :type value: object
        """

        if value is not None:
            assert hasattr(value, "__call__"), "'{0}' attribute: '{1}' is not callable!".format(
                "transfer_function", value)
        self.__transfer_function = value

    @property
    def inverse_transfer_function(self):
        """
        Property for **self.__inverse_transfer_function** attribute.

        :return: self.__inverse_transfer_function.
        :rtype: object
        """

        return self.__inverse_transfer_function

    @inverse_transfer_function.setter
    def inverse_transfer_function(self, value):
        """
        Setter for **self.__inverse_transfer_function** attribute.

        :param value: Attribute value.
        :type value: object
        """

        if value is not None:
            assert hasattr(value, "__call__"), "'{0}' attribute: '{1}' is not callable!".format(
                "inverse_transfer_function", value)
        self.__inverse_transfer_function = value


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE XYZ* colourspace to *RGB* colourspace using given *CIE XYZ* colourspace matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> XYZ = np.array([0.1151847498, 0.1008, 0.0508937252])
        >>> illuminant_XYZ =  (0.34567, 0.35850)
        >>> illuminant_RGB =  (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ =  np.array([3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3))
        >>> XYZ_to_RGB(XYZ, illuminant_XYZ, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        array([[ 0.17303501],
               [ 0.08211033],
               [ 0.05672498]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant_XYZ: *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: array_like
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: array_like
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode  ("XYZ Scaling", "Bradford", "Von Kries", "CAT02")
    :param from_XYZ: *Normalised primary matrix*.
    :type from_XYZ: array_like (3, 3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    :note: Input *illuminant_XYZ* is in domain [0, 1].
    :note: Input *illuminant_RGB* is in domain [0, 1].
    :note: Output *RGB* colourspace matrix is in domain [0, 1].
    """

    cat = colour.adaptation.cat.get_chromatic_adaptation_matrix(
        colour.models.cie_xyy.xy_to_XYZ(illuminant_XYZ),
        colour.models.cie_xyy.xy_to_XYZ(illuminant_RGB),
        method=chromatic_adaptation_method)

    adaptedXYZ = np.dot(cat, XYZ)

    RGB = np.dot(from_XYZ, adaptedXYZ)

    if transfer_function is not None:
        RGB = np.array(map(lambda x: transfer_function(x), np.ravel(RGB)))

    return RGB.reshape((3, 1))


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE XYZ* colourspace using given *RGB* colourspace matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> RGB = np.array([0.17303501, 0.08211033, 0.05672498])
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_XYZ = (0.34567, 0.35850)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> to_XYZ = np.array([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3))
        >>> RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, chromatic_adaptation_method, to_XYZ)
        array([[ 0.11518475],
               [ 0.1008    ],
               [ 0.05089373]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: array_like (3, 1)
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: array_like
    :param illuminant_XYZ: *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: array_like
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode  ("XYZ Scaling", "Bradford", "Von Kries", "CAT02")
    :param to_XYZ: *Normalised primary matrix*.
    :type to_XYZ: array_like (3, 3)
    :param inverse_transfer_function: *Inverse transfer function*.
    :type inverse_transfer_function: object
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *RGB* colourspace matrix is in domain [0, 1].
    :note: Input *illuminant_RGB* is in domain [0, 1].
    :note: Input *illuminant_XYZ* is in domain [0, 1].
    :note: Output *CIE XYZ* colourspace matrix is in domain [0, 1].
    """

    if inverse_transfer_function is not None:
        RGB = np.array(map(lambda x: inverse_transfer_function(x), np.ravel(RGB))).reshape((3, 1))

    XYZ = np.dot(to_XYZ, RGB)

    cat = colour.adaptation.cat.get_chromatic_adaptation_matrix(
        colour.models.cie_xyy.xy_to_XYZ(illuminant_RGB),
        colour.models.cie_xyy.xy_to_XYZ(illuminant_XYZ),
        method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ)

    return adapted_XYZ


def xyY_to_RGB(xyY,
               illuminant_xyY,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE xyY* colourspace to *RGB* colourspace using given *CIE xyY* colourspace matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> xyY = np.array([0.4316, 0.3777, 10.08])
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ = np.array([ 3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3)))
        >>> xyY_to_RGB(xyY, illuminant_xyY, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        array([[ 17.30350095],
               [  8.21103314],
               [  5.67249761]])

    :param xyY: *CIE xyY* colourspace matrix.
    :type xyY: array_like (3, 1)
    :param illuminant_xyY: *CIE xyY* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: array_like
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode  ("XYZ Scaling", "Bradford", "Von Kries", "CAT02")
    :param from_XYZ: *Normalised primary matrix*.
    :type from_XYZ: array_like (3, 3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *CIE xyY* colourspace matrix is in domain [0, 1].
    :note: Input *illuminant_xyY* is in domain [0, 1].
    :note: Input *illuminant_RGB* is in domain [0, 1].
    :note: Output *RGB* colourspace matrix is in domain [0, 1].
    """

    return XYZ_to_RGB(colour.models.cie_xyy.xyY_to_XYZ(xyY),
                      illuminant_xyY,
                      illuminant_RGB,
                      chromatic_adaptation_method,
                      from_XYZ,
                      transfer_function)


def RGB_to_xyY(RGB,
               illuminant_RGB,
               illuminant_xyY,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE xyY* colourspace using given *RGB* colourspace matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> RGB = np.array([17.303501, 8.211033, 5.672498])
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> chromatic_adaptation_method = "Bradford"
        >>> to_XYZ = np.array([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3)))
        >>> RGB_to_xyY(RGB, illuminant_RGB, illuminant_xyY, chromatic_adaptation_method, to_XYZ)
        array([[  0.4316    ],
               [  0.37769999],
               [ 10.0799999 ]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: array_like (3, 1)
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: array_like
    :param illuminant_xyY: *CIE xyY* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode ("XYZ Scaling", "Bradford", "Von Kries", "CAT02")
    :param to_XYZ: *Normalised primary* matrix.
    :type to_XYZ: array_like (3, 3)
    :param inverse_transfer_function: *Inverse transfer* function.
    :type inverse_transfer_function: object
    :return: *CIE xyY* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *RGB* colourspace matrix is in domain [0, 1].
    :note: Input *illuminant_RGB* is in domain [0, 1].
    :note: Input *illuminant_xyY* is in domain [0, 1].
    :note: Output *CIE xyY* is in domain [0, 1].
    """

    return colour.models.cie_xyy.XYZ_to_xyY(RGB_to_XYZ(RGB,
                                                             illuminant_RGB,
                                                             illuminant_xyY,
                                                             chromatic_adaptation_method,
                                                             to_XYZ,
                                                             inverse_transfer_function))


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_method="CAT02"):
    """
    Converts from given input *RGB* colourspace to output *RGB* colourspace using given *chromatic adaptation* method.

    Usage::

        >>> RGB = np.array([0.35521588, 0.41, 0.24177934])
        >>> RGB_to_RGB(RGB, colour.sRGB_COLOURSPACE, colour.PROPHOTO_RGB_COLOURSPACE)
        array([[ 0.35735427],
               [ 0.39987346],
               [ 0.26348887]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: array_like (3, 1)
    :param input_colourspace: *RGB* input colourspace.
    :type input_colourspace: RGB_Colourspace
    :param output_colourspace: *RGB* output colourspace.
    :type output_colourspace: RGB_Colourspace
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode  ("XYZ Scaling", "Bradford", "Von Kries", "CAT02")
    :return: *RGB* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *RGB* colourspace matrices are in domain [0, 1].
    """

    cat = colour.adaptation.cat.get_chromatic_adaptation_matrix(
        colour.models.cie_xyy.xy_to_XYZ(input_colourspace.whitepoint),
        colour.models.cie_xyy.xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_method)

    trs_matrix = np.dot(output_colourspace.from_XYZ, np.dot(cat, input_colourspace.to_XYZ))

    return np.dot(trs_matrix, RGB).reshape((3, 1))