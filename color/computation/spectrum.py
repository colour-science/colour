# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**spectrum.py**

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

import color.algebra.common
import color.utilities.exceptions
import color.utilities.verbose
from color.algebra.interpolation import SpragueInterpolator

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["SpectralPowerDistribution",
           "SpectralPowerDistributionTriad"]

LOGGER = color.utilities.verbose.install_logger()


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

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "wavelengths"))

    @wavelengths.deleter
    def wavelengths(self):
        """
        Deleter for **self.__wavelengths** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "values"))

    @values.deleter
    def values(self):
        """
        Deleter for **self.__values** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "values"))

    @property
    def shape(self):
        """
        Property for **self.__shape** attribute.

        :return: self.__shape.
        :rtype: tuple
        """

        steps = color.algebra.common.get_steps(self.wavelengths)
        return min(self.spd.keys()), max(self.spd.keys()), min(steps)

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.__shape** attribute.

        :param value: Attribute value.
        :type value: tuple
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "shape"))

    @shape.deleter
    def shape(self):
        """
        Deleter for **self.__shape** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

    def __eq__(self, spd):
        """
        Reimplements the :meth:`object.__eq__` method.

        :param spd: Spectral power distribution to compare for equality.
        :type spd: SpectralPowerDistribution
        :return: Spectral power distribution equality.
        :rtype: bool
        """

        for wavelength, value in self:
            if value != spd.get(wavelength):
                return False

        return True

    def __ne__(self, spd):
        """
        Reimplements the :meth:`object.__ne__` method.

        :param spd: Spectral power distribution to compare for inequality.
        :type spd: SpectralPowerDistribution
        :return: Spectral power distribution inequality.
        :rtype: bool
        """

        return not (self == spd)

    def __add__(self, x):
        """
        Reimplements the :meth:`object.__add__` method.

        :param x: Variable to add.
        :type x: float or ndarray
        :return: Variable added spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__spd = dict(zip(self.wavelengths, self.values + x))

        return self

    def __sub__(self, x):
        """
        Reimplements the :meth:`object.__sub__` method.

        :param x: Variable to subtract.
        :type x: float or ndarray
        :return: Variable subtracted spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        return self + (-x)

    def __mul__(self, x):
        """
        Reimplements the :meth:`object.__mul__` method.

        :param x: Variable to multiply.
        :type x: float or ndarray
        :return: Variable multiplied spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__spd = dict(zip(self.wavelengths, self.values * x))

        return self

    def __div__(self, x):
        """
        Reimplements the :meth:`object.__div__` method.

        :param x: Variable to divide.
        :type x: float or ndarray
        :return: Variable divided spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__spd = dict(zip(self.wavelengths, self.values * (1. / x)))

        return self

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

        return color.algebra.common.is_uniform(self.wavelengths)

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
        Interpolates the spectral power distribution following *CIE 167:2005* recommendations: the method developed
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
                spline_interpolant, spline_interpolator = linear_interpolant, None

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
                    raise color.utilities.exceptions.ProgrammingError(
                        "{0} | 'Sprague' interpolator can only be used for interpolating functions having a uniformly spaced independent variable!".format(
                            self.__class__.__name__))
            elif interpolator == "Cubic Spline":
                interpolant = spline_interpolant
            elif interpolator == "Linear":
                interpolant = linear_interpolant
            else:
                raise color.utilities.exceptions.ProgrammingError(
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

    def normalize(self, factor=1.):
        """
        Normalizes the spectral power distribution with given normalization factor.

        :param factor: Normalization factor
        :type factor: float
        :return: Normalized spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        return (self * (1. / max(self.values))) * factor

    def clone(self):
        """
        Clones the spectral power distribution.

        :return: Cloned spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        return copy.deepcopy(self)


class SpectralPowerDistributionTriad(object):
    """
    Defines a spectral power distribution triad implementation.
    """

    def __init__(self, name, triad, mapping, labels):
        """
        Initializes the class.

        :param name: Spectral power distribution triad name.
        :type name: str or unicode
        :param triad: Spectral power distribution triad.
        :type triad: dict
        :param mapping: Spectral power distribution triad attributes mapping.
        :type mapping: dict
        :param labels: Spectral power distribution triad axis labels mapping.
        :type labels: dict
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__mapping = mapping
        self.__triad = None
        self.triad = triad
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

        raise color.utilities.exceptions.ProgrammingError(
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
            for axis in ("x", "y", "z"):
                assert axis in value.keys(), \
                    "'{0}' attribute: '{1}' axis label is missing!".format("mapping", axis)
        self.__mapping = value

    @mapping.deleter
    def mapping(self):
        """
        Deleter for **self.__mapping** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "mapping"))

    @property
    def triad(self):
        """
        Property for **self.__triad** attribute.

        :return: self.__triad.
        :rtype: dict
        """

        return self.__triad

    @triad.setter
    def triad(self, value):
        """
        Setter for **self.__triad** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format("triad", value)
            for axis in ("x", "y", "z"):
                assert self.__mapping.get(axis) in value.keys(), \
                    "'{0}' attribute: '{1}' axis is missing!".format("triad", axis)

            triad = {}
            for axis in ("x", "y", "z"):
                triad[axis] = SpectralPowerDistribution(self.__mapping.get(axis), value.get(self.__mapping.get(axis)))

            numpy.testing.assert_almost_equal(triad["x"].wavelengths,
                                              triad["y"].wavelengths,
                                              err_msg="'{0}' attribute: '{1}' and '{2}' wavelengths are different!".format(
                                                  "triad", self.__mapping.get("x"), self.__mapping.get("y")))
            numpy.testing.assert_almost_equal(triad["x"].wavelengths,
                                              triad["z"].wavelengths,
                                              err_msg="'{0}' attribute: '{1}' and '{2}' wavelengths are different!".format(
                                                  "triad", self.__mapping.get("x"), self.__mapping.get("z")))

            self.__triad = triad
        else:
            self.__triad = None

    @triad.deleter
    def triad(self):
        """
        Deleter for **self.__triad** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "triad"))

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
            for axis in ("x", "y", "z"):
                assert axis in value.keys(), \
                    "'{0}' attribute: '{1}' axis label is missing!".format("labels", axis)
        self.__labels = value

    @labels.deleter
    def labels(self):
        """
        Deleter for **self.__labels** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "labels"))

    @property
    def x(self):
        """
        Property for **self.__x** attribute.

        :return: self.__x.
        :rtype: unicode
        """

        return self.__triad.get("x")

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x"))

    @x.deleter
    def x(self):
        """
        Deleter for **self.__x** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "x"))

    @property
    def y(self):
        """
        Property for **self.__y** attribute.

        :return: self.__y.
        :rtype: unicode
        """

        return self.__triad.get("y")

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y"))

    @y.deleter
    def y(self):
        """
        Deleter for **self.__y** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "y"))

    @property
    def z(self):
        """
        Property for **self.__z** attribute.

        :return: self.__z.
        :rtype: unicode
        """

        return self.__triad.get("z")

    @z.setter
    def z(self, value):
        """
        Setter for **self.__z** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z"))

    @z.deleter
    def z(self):
        """
        Deleter for **self.__z** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "wavelengths"))

    @wavelengths.deleter
    def wavelengths(self):
        """
        Deleter for **self.__wavelengths** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "values"))

    @values.deleter
    def values(self):
        """
        Deleter for **self.__values** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "shape"))

    @shape.deleter
    def shape(self):
        """
        Deleter for **self.__shape** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
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

        return itertools.izip(self.wavelengths, self.values)

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

    def __eq__(self, triad):
        """
        Reimplements the :meth:`object.__eq__` method.

        :param triad: Spectral power distribution triad to compare for equality.
        :type triad: SpectralPowerDistributionTriad
        :return: Spectral power distribution triad equality.
        :rtype: bool
        """

        equality = True
        for axis in self.__mapping:
            equality *= getattr(self, axis) == getattr(triad, axis)

        return equality

    def __ne__(self, triad):
        """
        Reimplements the :meth:`object.__eq__` method.

        :param triad: Spectral power distribution triad to compare for inequality.
        :type triad: SpectralPowerDistributionTriad
        :return: Spectral power distribution triad inequality.
        :rtype: bool
        """

        return not (self == triad)

    def __add__(self, x):
        """
        Reimplements the :meth:`object.__add__` method.

        :param x: Variable to add.
        :type x: float or ndarray
        :return: Variable added spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        values = self.values + x

        for i, axis in enumerate(("x", "y", "z")):
            self.__triad[axis].spd = dict(zip(self.wavelengths, values[:, i]))

        return self

    def __sub__(self, x):
        """
        Reimplements the :meth:`object.__sub__` method.

        :param x: Variable to subtract.
        :type x: float or ndarray
        :return: Variable subtracted spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        return self + (-x)

    def __mul__(self, x):
        """
        Reimplements the :meth:`object.__mul__` method.

        :param x: Variable to multiply.
        :type x: float or ndarray
        :return: Variable multiplied spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        values = self.values * x

        for i, axis in enumerate(("x", "y", "z")):
            self.__triad[axis].spd = dict(zip(self.wavelengths, values[:, i]))

        return self

    def __div__(self, x):
        """
        Reimplements the :meth:`object.__div__` method.

        :param x: Variable to divide.
        :type x: float or ndarray
        :return: Variable divided spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        return self * (1. / x)

    def is_uniform(self):
        """
        Returns if the spectral power distribution triad have uniformly spaced data.

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
        Extrapolates the spectral power distribution triad according to *CIE 15:2004* recommendation.

        Reference: https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf, 7.2.2.1 Extrapolation

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :return: Extrapolated spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        for i in self.__mapping.keys():
            getattr(self, i).extrapolate(start, end)

        return self

    def interpolate(self, start=None, end=None, steps=None):
        """
        Interpolates the spectral power distribution triad following *CIE 167:2005* recommendations.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Interpolated spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)

        return self

    def align(self, start, end, steps):
        """
        Aligns the spectral power distribution triad to given shape: Interpolates first then extrapolates to fit the given range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Aligned spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)
            getattr(self, i).extrapolate(start, end)

        return self

    def zeros(self, start=None, end=None, steps=None):
        """
        Zeros fills the spectral power distribution triad: Missing values will be replaced with zeros to fit the defined range.

        :param start: Wavelengths range start.
        :type start: float
        :param end: Wavelengths range end.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Zeros filled spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        for i in self.__mapping.keys():
            getattr(self, i).zeros(start, end, steps)

        return self

    def normalize(self, factor=1.):
        """
        Normalizes the spectral power distribution triad with given normalization factor.

        :param factor: Normalization factor
        :type factor: float
        :return: Normalized spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        for i in self.__mapping.keys():
            getattr(self, i).normalize(factor)

        return self

    def clone(self):
        """
        Clones the spectral power distribution triad.

        :return: Cloned spectral power distribution triad.
        :rtype: SpectralPowerDistributionTriad
        """

        return copy.deepcopy(self)