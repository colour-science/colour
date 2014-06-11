# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**spd.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *spectral power distribution* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import copy
import itertools
import math
import numpy

import color.exceptions
import color.utilities.common
import color.verbose
from color.algebra.interpolation import SpragueInterpolator

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "SpectralPowerDistribution",
           "AbstractColorMatchingFunctions",
           "RGB_ColorMatchingFunctions",
           "XYZ_ColorMatchingFunctions"]

LOGGER = color.verbose.install_logger()


class SpectralPowerDistribution(object):
    """
    Defines a spectral power distribution object.
    """

    def __init__(self, name, spd):
        """
        Initializes the class.

        :param name: Spectral power distribution name.
        :type name: str or unicode
        :param spd: Spectral power distribution data.
        :type spd: dict
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__spd = None
        self.spd = spd

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
    def spd(self):
        """
        Property for **self.__spd** attribute.

        :return: self.__spd.
        :rtype: dict
        """

        return self.__spd

    @spd.setter
    def spd(self, value):
        """
        Setter for **self.__spd** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format("spd",
                                                                                            value)
        self.__spd = value

    @spd.deleter
    def spd(self):
        """
        Deleter for **self.__spd** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "spd"))

    @property
    def wavelengths(self):
        """
        Property for **self.__wavelengths** attribute.

        :return: self.__wavelengths.
        :rtype: list
        """

        return numpy.array(sorted(self.__spd.keys()))

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.__wavelengths** attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "wavelengths"))

    @wavelengths.deleter
    def wavelengths(self):
        """
        Deleter for **self.__wavelengths** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "wavelengths"))

    @property
    def values(self):
        """
        Property for **self.__values** attribute.

        :return: self.__values.
        :rtype: list
        """

        return numpy.array(map(self.get, self.wavelengths))

    @values.setter
    def values(self, value):
        """
        Setter for **self.__values** attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "values"))

    @values.deleter
    def values(self):
        """
        Deleter for **self.__values** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "values"))

    @property
    def shape(self):
        """
        Property for **self.__shape** attribute.

        :return: self.__shape.
        :rtype: tuple
        """

        steps = color.utilities.common.get_steps(self.wavelengths)
        return min(self.spd.keys()), max(self.spd.keys()), min(steps)

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.__shape** attribute.

        :param value: Attribute value.
        :type value: tuple
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "shape"))

    @shape.deleter
    def shape(self):
        """
        Deleter for **self.__shape** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "shape"))

    def __getitem__(self, wavelength):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Value.
        :rtype: float
        """

        return self.__spd.__getitem__(wavelength)

    def __setitem__(self, key, value):
        """
        Reimplements the :meth:`object.__setitem__` method.

        :param key: Key.
        :type key: unicode
        :param value: Value.
        :type value: object
        """

        self.__spd.__setitem__(key, value)

    def __iter__(self):
        """
        Reimplements the :meth:`object.__iter__` method.

        :return: Spectral power distribution iterator.
        :rtype: object
        """

        return iter(sorted(self.__spd.items()))

    def __contains__(self, wavelength):
        """
        Reimplements the :meth:`object.__contains__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Wavelength existence.
        :rtype: bool
        """

        return wavelength in self.__spd

    def __len__(self):
        """
        Reimplements the :meth:`object.__len__` method.

        :return: Wavelengths count.
        :rtype: int
        """

        return len(self.__spd)

    def get(self, wavelength, default=None):
        """
        Returns given wavelength value.

        :param wavelength: Wavelength.
        :type wavelength: float
        :param default: Default value if wavelength is not found.
        :type default: object
        :return: Value.
        :rtype: float
        """

        try:
            return self.__getitem__(wavelength)
        except KeyError as error:
            return default

    def is_uniform(self):
        """
        Returns if the spectral power distribution has uniformly spaced data.

        :return: Is uniform.
        :rtype: bool
        """

        return color.utilities.common.is_uniform(self.wavelengths)

    def extrapolate(self, start, end):
        """
        Extrapolates the spectral power distribution according to *CIE 15:2004* recommendation.

        Reference: https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf, 7.2.2.1 Extrapolation

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :return: Extrapolated spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        start_wavelength, end_wavelength, steps = self.shape

        minimum, maximum = self.get(start_wavelength), self.get(end_wavelength)
        for i in numpy.arange(start_wavelength, start - steps, -steps):
            self[i] = minimum
        for i in numpy.arange(end_wavelength, end + steps, steps):
            self[i] = maximum

        return self

    def interpolate(self, start=None, end=None, steps=None, interpolator=None):
        """
        Interpolates the spectral power distribution following *CIE* recommendations: the method developed
        by *Sprague* (1880) should be used for interpolating functions having a uniformly spaced independent variable and
        a *Cubic Spline* method for non-uniformly spaced independent variable.

        Reference: http://div1.cie.co.at/?i_ca_id=551&pubid=47

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :param interpolator: Interpolator to enforce usage.
        :type interpolator: unicode ("Sprague", "Cubic Spline", "Linear")
        :return: Interpolated spectral power distribution.
        :rtype: SpectralPowerDistribution

        :note: *Sprague* interpolator cannot be used for interpolating functions having a non-uniformly spaced independent variable.
        :note: If *Scipy* is not unavailable the *Cubic Spline* method will fallback to legacy *Linear* interpolation.
        """

        shape_start, shape_end, shape_steps = self.shape
        start, end, steps = map(lambda x: x[0] if x[0] is not None else x[1], zip((start, end, steps),
                                                                                  (
                                                                                      shape_start, shape_end,
                                                                                      shape_steps)))

        if shape_steps != steps:
            wavelengths, values = self.wavelengths, self.values

            is_uniform = self.is_uniform()
            # Initialising *Sprague* interpolant.
            if is_uniform:
                sprague_interpolator = SpragueInterpolator(wavelengths, values)
                sprague_interpolant = lambda x: sprague_interpolator(x)
            else:
                sprague_interpolant = sprague_interpolator = None

            # Initialising *Linear* interpolant.
            linear_interpolant = lambda x: numpy.interp(x, wavelengths, values)

            # Initialising *Cubic Spline* interpolant.
            try:
                from scipy.interpolate import interp1d

                spline_interpolator = interp1d(wavelengths, values, kind="cubic")
                spline_interpolant = lambda x: spline_interpolator(x)
            except ImportError as error:
                LOGGER.warning(
                    "!> {0} | 'scipy.interpolate.interp1d' interpolator is unavailable, using 'numpy.interp' interpolator!".format(
                        self.__class__.__name__))
                spline_interpolant,  spline_interpolator = linear_interpolant, None

            # Defining proper interpolation bounds.
            # TODO: Provide support for fractional steps like 0.1, etc...
            shape_start, shape_end = math.ceil(shape_start), math.floor(shape_end)

            if interpolator is None:
                if is_uniform:
                    interpolant = sprague_interpolant
                else:
                    interpolant = spline_interpolant
            elif interpolator == "Sprague":
                if is_uniform:
                    interpolant = sprague_interpolant
                else:
                    raise color.exceptions.ProgrammingError(
                        "{0} | 'Sprague' interpolator can only be used for interpolating functions having a uniformly spaced independent variable!".format(
                            self.__class__.__name__))
            elif interpolator == "Cubic Spline":
                interpolant = spline_interpolant
            elif interpolator == "Linear":
                interpolant = linear_interpolant
            else:
                raise color.exceptions.ProgrammingError(
                        "{0} | Undefined '{1}' interpolator!".format(self.__class__.__name__, interpolator))

            LOGGER.debug(
                "> {0} | Interpolated '{1}' spectral power distribution shape: {2}.".format(self.__class__.__name__,
                                                                                          self.name,
                                                                                          (shape_start, shape_end,
                                                                                           steps)))

            self.__spd = dict([(wavelength, interpolant(wavelength))
                               for wavelength in numpy.arange(max(start, shape_start),
                                                              min(end, shape_end) + steps,
                                                              steps)])
        return self

    def align(self, start, end, steps):
        """
        Aligns the spectral power distribution to given shape: Interpolates first then extrapolates to fit the given range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Aligned spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.interpolate(start, end, steps)
        self.extrapolate(start, end)

        return self

    def zeros(self, start=None, end=None, steps=None):
        """
        Zeros fills the spectral power distribution: Missing values will be replaced with zeroes to fit the defined range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Zeros filled spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        start, end, steps = map(lambda x: x[0] if x[0] is not None else x[1], zip((start, end, steps), self.shape))

        self.__spd = dict(
            [(wavelength, self.get(wavelength, 0.)) for wavelength in numpy.arange(start, end + steps, steps)])

        return self

    def clone(self):
        """
        Clones the spectral power distribution.

        :return: Cloned spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        return copy.deepcopy(self)

class AbstractColorMatchingFunctions(object):
    """
    Defines an abstract standard observer color matching functions object implementation.
    """

    def __init__(self, name, cmfs, mapping, labels):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: str or unicode
        :param cmfs: Standard observer color matching functions.
        :type cmfs: dict
        :param mapping: Standard observer color matching functions attributes mapping.
        :type mapping: dict
        :param labels: Standard observer color matching functions axis labels mapping.
        :type labels: dict
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__mapping = mapping
        self.__cmfs = None
        self.cmfs = cmfs
        self.__labels = labels

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
    def mapping(self):
        """
        Property for **self.__mapping** attribute.

        :return: self.__mapping.
        :rtype: dict
        """

        return self.__mapping

    @mapping.setter
    def mapping(self, value):
        """
        Setter for **self.__mapping** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format("mapping", value)
            for cmf in ("x", "y", "z"):
                assert cmf in value.keys(), \
                    "'{0}' attribute: '{1}' matching function label is missing!".format("mapping", cmf)
        self.__mapping = value

    @mapping.deleter
    def mapping(self):
        """
        Deleter for **self.__mapping** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "mapping"))

    @property
    def cmfs(self):
        """
        Property for **self.__cmfs** attribute.

        :return: self.__cmfs.
        :rtype: dict
        """

        return self.__cmfs

    @cmfs.setter
    def cmfs(self, value):
        """
        Setter for **self.__cmfs** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format("cmfs", value)
            for cmf in ("x", "y", "z"):
                assert self.__mapping.get(cmf) in value.keys(), \
                    "'{0}' attribute: '{1}' matching function is missing!".format("cmfs", cmf)

            cmfs = {}

            cmfs["x"] = SpectralPowerDistribution(self.__mapping.get("x"), value.get(self.__mapping.get("x")))
            cmfs["y"] = SpectralPowerDistribution(self.__mapping.get("y"), value.get(self.__mapping.get("y")))
            cmfs["z"] = SpectralPowerDistribution(self.__mapping.get("z"), value.get(self.__mapping.get("z")))

            numpy.testing.assert_almost_equal(cmfs["x"].wavelengths,
                                              cmfs["y"].wavelengths,
                                              err_msg="'{0}' attribute: '{1}' and '{2}' matching function wavelengths are different!".format(
                                                  "cmfs", self.__mapping.get("x"), self.__mapping.get("y")))
            numpy.testing.assert_almost_equal(cmfs["x"].wavelengths,
                                              cmfs["z"].wavelengths,
                                              err_msg="'{0}' attribute: '{1}' and '{2}' matching function wavelengths are different!".format(
                                                  "cmfs", self.__mapping.get("x"), self.__mapping.get("z")))

            self.__cmfs = cmfs
        else:
            self.__cmfs = None

    @cmfs.deleter
    def cmfs(self):
        """
        Deleter for **self.__cmfs** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "cmfs"))

    @property
    def labels(self):
        """
        Property for **self.__labels** attribute.

        :return: self.__labels.
        :rtype: dict
        """

        return self.__labels

    @labels.setter
    def labels(self, value):
        """
        Setter for **self.__labels** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format("labels", value)
            for cmf in ("x", "y", "z"):
                assert cmf in value.keys(), \
                    "'{0}' attribute: '{1}' matching function label is missing!".format("labels", cmf)
        self.__labels = value

    @labels.deleter
    def labels(self):
        """
        Deleter for **self.__labels** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "labels"))

    @property
    def x(self):
        """
        Property for **self.__x** attribute.

        :return: self.__x.
        :rtype: unicode
        """

        return self.__cmfs.get("x")

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x"))

    @x.deleter
    def x(self):
        """
        Deleter for **self.__x** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "x"))

    @property
    def y(self):
        """
        Property for **self.__y** attribute.

        :return: self.__y.
        :rtype: unicode
        """

        return self.__cmfs.get("y")

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y"))

    @y.deleter
    def y(self):
        """
        Deleter for **self.__y** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "y"))

    @property
    def z(self):
        """
        Property for **self.__z** attribute.

        :return: self.__z.
        :rtype: unicode
        """

        return self.__cmfs.get("z")

    @z.setter
    def z(self, value):
        """
        Setter for **self.__z** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z"))

    @z.deleter
    def z(self):
        """
        Deleter for **self.__z** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "z"))

    @property
    def wavelengths(self):
        """
        Property for **self.__wavelengths** attribute.

        :return: self.__wavelengths.
        :rtype: list
        """

        return self.x.wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.__wavelengths** attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "wavelengths"))

    @wavelengths.deleter
    def wavelengths(self):
        """
        Deleter for **self.__wavelengths** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "wavelengths"))

    @property
    def values(self):
        """
        Property for **self.__values** attribute.

        :return: self.__values.
        :rtype: list
        """

        return numpy.array(map(self.get, self.wavelengths))

    @values.setter
    def values(self, value):
        """
        Setter for **self.__values** attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "values"))

    @values.deleter
    def values(self):
        """
        Deleter for **self.__values** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "values"))

    @property
    def shape(self):
        """
        Property for **self.__shape** attribute.

        :return: self.__shape.
        :rtype: tuple
        """

        return self.x.shape

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.__shape** attribute.

        :param value: Attribute value.
        :type value: tuple
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "shape"))

    @shape.deleter
    def shape(self):
        """
        Deleter for **self.__shape** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "shape"))

    def __getitem__(self, wavelength):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Value.
        :rtype: ndarray
        """

        return numpy.array((self.x[wavelength], self.y[wavelength], self.z[wavelength]))

    def __setitem__(self, key, value):
        """
        Reimplements the :meth:`object.__setitem__` method.

        :param key: Key.
        :type key: unicode
        :param value: Value.
        :type value: tuple
        """

        x, y, z = value

        self.x.__setitem__(key, x)
        self.y.__setitem__(key, y)
        self.z.__setitem__(key, z)

    def __iter__(self):
        """
        Reimplements the :meth:`object.__iter__` method.

        :return: Spectral distribution iterator.
        :rtype: object
        """

        return itertools.izip(self.wavelengths,
                              zip(*([value for key, value in self.x],
                                    [value for key, value in self.y],
                                    [value for key, value in self.z])))

    def __contains__(self, wavelength):
        """
        Reimplements the :meth:`object.__contains__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Wavelength existence.
        :rtype: bool
        """

        return wavelength in self.x

    def __len__(self):
        """
        Reimplements the :meth:`object.__len__` method.

        :return: Wavelengths count.
        :rtype: int
        """

        return len(self.x)

    def is_uniform(self):
        """
        Returns if the color matching functions have uniformly spaced data.

        :return: Is uniform.
        :rtype: bool
        """

        for i in self.__mapping.keys():
            if not getattr(self, i).is_uniform():
                return False
        return True

    def get(self, wavelength, default=None):
        """
        Returns given wavelength value.

        :param wavelength: Wavelength.
        :type wavelength: float
        :param default: Default value if wavelength is not found.
        :type default: object
        :return: Value.
        :rtype: float
        """

        try:
            return self.__getitem__(wavelength)
        except KeyError as error:
            return default

    def extrapolate(self, start=None, end=None):
        """
        Extrapolates the color matching functions according to *CIE 15:2004* recommendation.

        Reference: https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf, 7.2.2.1 Extrapolation

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :return: Extrapolated color matching functions.
        :rtype: AbstractColorMatchingFunctions
        """

        for i in self.__mapping.keys():
            getattr(self, i).extrapolate(start, end)

        return self

    def interpolate(self, start=None, end=None, steps=None):
        """
        Interpolates the color matching functions following *CIE* recommendations.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Interpolated color matching functions.
        :rtype: AbstractColorMatchingFunctions
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)

        return self

    def align(self, start, end, steps):
        """
        Aligns the color matching functions to given shape: Interpolates first then extrapolates to fit the given range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Aligned color matching functions.
        :rtype: AbstractColorMatchingFunctions
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)
            getattr(self, i).extrapolate(start, end)

        return self

    def zeros(self, start=None, end=None, steps=None):
        """
        Zeros fills the color matching functions: Missing values will be replaced with zeros to fit the defined range.

        :param start: Wavelengths range start.
        :type start: float
        :param end: Wavelengths range end.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Zeros filled color matching functions.
        :rtype: AbstractColorMatchingFunctions
        """

        for i in self.__mapping.keys():
            getattr(self, i).zeros(start, end, steps)

        return self

    def clone(self):
        """
        Clones the color matching functions.

        :return: Cloned color matching functions.
        :rtype: AbstractColorMatchingFunctions
        """

        return copy.deepcopy(self)

class RGB_ColorMatchingFunctions(AbstractColorMatchingFunctions):
    """
    Defines a *CIE RGB* standard observer color matching functions object implementation.
    """

    def __init__(self, name, cmfs):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: unicode
        :param cmfs: Standard observer color matching functions.
        :type cmfs: dict
        """

        AbstractColorMatchingFunctions.__init__(self,
                                                name,
                                                cmfs,
                                                mapping={"x": "r_bar",
                                                         "y": "g_bar",
                                                         "z": "b_bar"},
                                                labels={"x": "r\u0304",
                                                        "y": "g\u0304",
                                                        "z": "b\u0304"})

    @property
    def r_bar(self):
        """
        Property for **self.__r_bar** attribute.

        :return: self.__r_bar.
        :rtype: unicode
        """

        return self.x

    @r_bar.setter
    def r_bar(self, value):
        """
        Setter for **self.__r_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "r_bar"))

    @r_bar.deleter
    def r_bar(self):
        """
        Deleter for **self.__r_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "r_bar"))

    @property
    def g_bar(self):
        """
        Property for **self.__g_bar** attribute.

        :return: self.__g_bar.
        :rtype: unicode
        """

        return self.y

    @g_bar.setter
    def g_bar(self, value):
        """
        Setter for **self.__g_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "g_bar"))

    @g_bar.deleter
    def g_bar(self):
        """
        Deleter for **self.__g_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "g_bar"))

    @property
    def b_bar(self):
        """
        Property for **self.__b_bar** attribute.

        :return: self.__b_bar.
        :rtype: unicode
        """

        return self.z

    @b_bar.setter
    def b_bar(self, value):
        """
        Setter for **self.__b_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "b_bar"))

    @b_bar.deleter
    def b_bar(self):
        """
        Deleter for **self.__b_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "b_bar"))


class XYZ_ColorMatchingFunctions(AbstractColorMatchingFunctions):
    """
    Defines an *CIE XYZ* standard observer color matching functions object implementation.
    """

    def __init__(self, name, cmfs):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: unicode
        :param cmfs: Standard observer color matching functions.
        :type cmfs: dict
        """

        AbstractColorMatchingFunctions.__init__(self,
                                                name,
                                                cmfs,
                                                mapping={"x": "x_bar",
                                                         "y": "y_bar",
                                                         "z": "z_bar"},
                                                labels={"x": "x\u0304",
                                                        "y": "y\u0304",
                                                        "z": "z\u0304"})

    @property
    def x_bar(self):
        """
        Property for **self.__x_bar** attribute.

        :return: self.__x_bar.
        :rtype: unicode
        """

        return self.x

    @x_bar.setter
    def x_bar(self, value):
        """
        Setter for **self.__x_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x_bar"))

    @x_bar.deleter
    def x_bar(self):
        """
        Deleter for **self.__x_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "x_bar"))

    @property
    def y_bar(self):
        """
        Property for **self.__y_bar** attribute.

        :return: self.__y_bar.
        :rtype: unicode
        """

        return self.y

    @y_bar.setter
    def y_bar(self, value):
        """
        Setter for **self.__y_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y_bar"))

    @y_bar.deleter
    def y_bar(self):
        """
        Deleter for **self.__y_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "y_bar"))

    @property
    def z_bar(self):
        """
        Property for **self.__z_bar** attribute.

        :return: self.__z_bar.
        :rtype: unicode
        """

        return self.z

    @z_bar.setter
    def z_bar(self, value):
        """
        Setter for **self.__z_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z_bar"))

    @z_bar.deleter
    def z_bar(self):
        """
        Deleter for **self.__z_bar** attribute.
        """

        raise color.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "z_bar"))