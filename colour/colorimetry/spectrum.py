# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**spectrum.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *spectral power distribution* objects.

**Others:**

"""

from __future__ import unicode_literals

import copy
import itertools
import math
import numpy as np

from colour.algebra import is_uniform, get_steps
from colour.utilities import is_scipy_installed, warning
from colour.algebra.interpolation import LinearInterpolator1d
from colour.algebra.interpolation import SpragueInterpolator

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["SpectralPowerDistribution",
           "TriSpectralPowerDistribution"]


class SpectralPowerDistribution(object):
    """
    Defines a spectral power distribution object.
    """

    def __init__(self, name, data):
        """
        :param name: Spectral power distribution name.
        :type name: str or unicode
        :param data: Spectral power distribution data.
        :type data: dict
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__data = None
        self.data = data

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        :return: self.__name.
        :rtype: str or unicode
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        :param value: Attribute value.
        :type value: str or unicode
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                "'{0}' attribute: '{1}' type is not in 'str' or 'unicode'!".format(
                    "name", value)
        self.__name = value

    @property
    def data(self):
        """
        Property for **self.__data** private attribute.

        :return: self.__data.
        :rtype: dict
        """

        return self.__data

    @data.setter
    def data(self, value):
        """
        Setter for **self.__data** private attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, \
                "'{0}' attribute: '{1}' type is not 'dict'!".format(
                    "data", value)
        self.__data = value

    @property
    def wavelengths(self):
        """
        Property for **self.__wavelengths** private attribute.

        :return: self.__wavelengths.
        :rtype: list
        """

        return np.array(sorted(self.__data.keys()))

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.__wavelengths** private attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "wavelengths"))

    @property
    def values(self):
        """
        Property for **self.__values** private attribute.

        :return: self.__values.
        :rtype: list
        """

        return np.array([self.get(wavelength)
                         for wavelength in self.wavelengths])

    @values.setter
    def values(self, value):
        """
        Setter for **self.__values** private attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "values"))

    @property
    def shape(self):
        """
        Property for **self.__shape** private attribute.

        :return: self.__shape.
        :rtype: tuple
        """

        steps = get_steps(self.wavelengths)
        return min(self.data.keys()), max(self.data.keys()), min(steps)

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.__shape** private attribute.

        :param value: Attribute value.
        :type value: tuple
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "shape"))

    def __hash__(self):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :return: Object hash.
        :rtype: int
        """

        return hash(id(self))

    def __getitem__(self, wavelength):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Value.
        :rtype: float
        """

        return self.__data.__getitem__(wavelength)

    def __setitem__(self, key, value):
        """
        Reimplements the :meth:`object.__setitem__` method.

        :param key: Key.
        :type key: unicode
        :param value: Value.
        :type value: object
        """

        self.__data.__setitem__(key, value)

    def __iter__(self):
        """
        Reimplements the :meth:`object.__iter__` method.

        :return: Spectral power distribution iterator.
        :rtype: object
        """

        return iter(sorted(self.__data.items()))

    def __contains__(self, wavelength):
        """
        Reimplements the :meth:`object.__contains__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Wavelength existence.
        :rtype: bool
        """

        return wavelength in self.__data

    def __len__(self):
        """
        Reimplements the :meth:`object.__len__` method.

        :return: Wavelengths count.
        :rtype: int
        """

        return len(self.__data)

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
        :type x: float or array_like
        :return: Variable added spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__data = dict(tuple(zip(self.wavelengths, self.values + x)))

        return self

    def __sub__(self, x):
        """
        Reimplements the :meth:`object.__sub__` method.

        :param x: Variable to subtract.
        :type x: float or array_like
        :return: Variable subtracted spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        return self + (-x)

    def __mul__(self, x):
        """
        Reimplements the :meth:`object.__mul__` method.

        :param x: Variable to multiply.
        :type x: float or array_like
        :return: Variable multiplied spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__data = dict(tuple(zip(self.wavelengths, self.values * x)))

        return self

    def __div__(self, x):
        """
        Reimplements the :meth:`object.__div__` method.

        :param x: Variable to divide.
        :type x: float or array_like
        :return: Variable divided spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        self.__data = dict(
            tuple(zip(self.wavelengths, self.values * (1. / x))))

        return self

    # Python 3 compatibility.
    __truediv__ = __div__

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

        return is_uniform(self.wavelengths)

    def extrapolate(self, start, end):
        """
        Extrapolates the spectral power distribution according to
        *CIE 15:2004* recommendation.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :return: Extrapolated spectral power distribution.
        :rtype: SpectralPowerDistribution

        References:

        -  `CIE 015:2004 Colorimetry, 3rd edition: \
        7.2.2.1 Extrapolation \
        <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_
        -  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: \
        10. EXTRAPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        """

        start_wavelength, end_wavelength, steps = self.shape

        minimum, maximum = self.get(start_wavelength), self.get(end_wavelength)
        for i in np.arange(start_wavelength, start - steps, -steps):
            self[i] = minimum
        for i in np.arange(end_wavelength, end + steps, steps):
            self[i] = maximum

        return self

    def interpolate(self, start=None, end=None, steps=None, interpolator=None):
        """
        Interpolates the spectral power distribution following
        *CIE 167:2005* recommendations: the method developed by *Sprague* (1880)
        should be used for interpolating functions having a uniformly spaced
        independent variable and a *Cubic Spline* method for non-uniformly
        spaced independent variable.

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

        :note: *Sprague* interpolator cannot be used for interpolating \
        functions having a non-uniformly spaced independent variable.
        :note: If *scipy* is not unavailable the *Cubic Spline* method will \
        fallback to legacy *Linear* interpolation.

        References:

        -  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: \
        9. INTERPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        """

        shape_start, shape_end, shape_steps = self.shape
        boundaries = tuple(zip((start, end, steps),
                              (shape_start, shape_end, shape_steps)))
        start, end, steps = [x[0] if x[0] is not None else x[1]
                             for x in boundaries]

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
            linear_interpolator = LinearInterpolator1d(wavelengths, values)
            linear_interpolant = lambda x: linear_interpolator(x)

            # Initialising *Cubic Spline* interpolant.
            if is_scipy_installed():
                from scipy.interpolate import interp1d

                spline_interpolator = interp1d(wavelengths, values,
                                               kind="cubic")
                spline_interpolant = lambda x: spline_interpolator(x)
            else:
                warning(
                    "!> {0} | 'scipy.interpolate.interp1d' interpolator is unavailable, using 'np.interp' interpolator!".format(
                        self.__class__.__name__))
                spline_interpolant, spline_interpolator = linear_interpolant, None

            # Defining proper interpolation bounds.
            # TODO: Provide support for fractional steps like 0.1, etc...
            shape_start, shape_end = math.ceil(shape_start), math.floor(
                shape_end)

            if interpolator is None:
                if is_uniform:
                    interpolant = sprague_interpolant
                else:
                    interpolant = spline_interpolant
            elif interpolator == "Sprague":
                if is_uniform:
                    interpolant = sprague_interpolant
                else:
                    raise RuntimeError(
                        "{0} | 'Sprague' interpolator can only be used for interpolating functions having a uniformly spaced independent variable!".format(
                            self.__class__.__name__))
            elif interpolator == "Cubic Spline":
                interpolant = spline_interpolant
            elif interpolator == "Linear":
                interpolant = linear_interpolant
            else:
                raise ValueError("{0} | Undefined '{1}' interpolator!".format(
                    self.__class__.__name__, interpolator))

            self.__data = dict([(wavelength, interpolant(wavelength))
                                for wavelength in
                                np.arange(max(start, shape_start),
                                          min(end, shape_end) + steps,
                                          steps)])
        return self

    def align(self, start, end, steps):
        """
        Aligns the spectral power distribution to given shape: Interpolates
        first then extrapolates to fit the given range.

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
        Zeros fills the spectral power distribution: Missing values will be
        replaced with zeroes to fit the defined range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Zeros filled spectral power distribution.
        :rtype: SpectralPowerDistribution
        """

        start, end, steps = [x[0] if x[0] is not None else x[1]
                             for x in
                             tuple(zip((start, end, steps), self.shape))]

        self.__data = dict(
            [(wavelength, self.get(wavelength, 0.)) for wavelength in
             np.arange(start, end + steps, steps)])

        return self

    def normalise(self, factor=1.):
        """
        Normalises the spectral power distribution with given normalization
        factor.

        :param factor: Normalization factor
        :type factor: float
        :return: Normalised spectral power distribution.
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


class TriSpectralPowerDistribution(object):
    """
    Defines a tri-spectral power distribution implementation.
    """

    def __init__(self, name, data, mapping, labels):
        """
        :param name: Tri-spectral power distribution name.
        :type name: str or unicode
        :param data: Tri-spectral power distribution data.
        :type data: dict
        :param mapping: Tri-spectral power distribution attributes mapping.
        :type mapping: dict
        :param labels: Tri-spectral power distribution axis labels mapping.
        :type labels: dict
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__mapping = mapping
        self.__data = None
        self.data = data
        self.__labels = labels

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        :return: self.__name.
        :rtype: str or unicode
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        :param value: Attribute value.
        :type value: str or unicode
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                "'{0}' attribute: '{1}' type is not in 'str' or 'unicode'!".format(
                    "name", value)
        self.__name = value

    @property
    def mapping(self):
        """
        Property for **self.__mapping** private attribute.

        :return: self.__mapping.
        :rtype: dict
        """

        return self.__mapping

    @mapping.setter
    def mapping(self, value):
        """
        Setter for **self.__mapping** private attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) is dict, \
                "'{0}' attribute: '{1}' type is not 'dict'!".format(
                    "mapping", value)
            for axis in ("x", "y", "z"):
                assert axis in value.keys(), \
                    "'{0}' attribute: '{1}' axis label is missing!".format(
                        "mapping", axis)
        self.__mapping = value

    @property
    def data(self):
        """
        Property for **self.__data** private attribute.

        :return: self.__data.
        :rtype: dict
        """

        return self.__data

    @data.setter
    def data(self, value):
        """
        Setter for **self.__data** private attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(
                value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format(
                "data", value)
            for axis in ("x", "y", "z"):
                assert self.__mapping.get(axis) in value.keys(), \
                    "'{0}' attribute: '{1}' axis is missing!".format(
                        "data", axis)

            data = {}
            for axis in ("x", "y", "z"):
                data[axis] = SpectralPowerDistribution(
                    self.__mapping.get(axis),
                    value.get(self.__mapping.get(axis)))

            np.testing.assert_almost_equal(
                data["x"].wavelengths,
                data["y"].wavelengths,
                err_msg="'{0}' attribute: '{1}' and '{2}' wavelengths are different!".format(
                    "data", self.__mapping.get("x"),
                    self.__mapping.get("y")))
            np.testing.assert_almost_equal(
                data["x"].wavelengths,
                data["z"].wavelengths,
                err_msg="'{0}' attribute: '{1}' and '{2}' wavelengths are different!".format(
                    "data", self.__mapping.get("x"),
                    self.__mapping.get("z")))

            self.__data = data
        else:
            self.__data = None

    @property
    def labels(self):
        """
        Property for **self.__labels** private attribute.

        :return: self.__labels.
        :rtype: dict
        """

        return self.__labels

    @labels.setter
    def labels(self, value):
        """
        Setter for **self.__labels** private attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(
                value) is dict, "'{0}' attribute: '{1}' type is not 'dict'!".format(
                "labels", value)
            for axis in ("x", "y", "z"):
                assert axis in value.keys(), \
                    "'{0}' attribute: '{1}' axis label is missing!".format(
                        "labels", axis)
        self.__labels = value

    @property
    def x(self):
        """
        Property for **self.__x** private attribute.

        :return: self.__x.
        :rtype: unicode
        """

        return self.__data.get("x")

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** private attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "x"))

    @property
    def y(self):
        """
        Property for **self.__y** private attribute.

        :return: self.__y.
        :rtype: unicode
        """

        return self.__data.get("y")

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** private attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "y"))

    @property
    def z(self):
        """
        Property for **self.__z** private attribute.

        :return: self.__z.
        :rtype: unicode
        """

        return self.__data.get("z")

    @z.setter
    def z(self, value):
        """
        Setter for **self.__z** private attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "z"))

    @property
    def wavelengths(self):
        """
        Property for **self.__wavelengths** private attribute.

        :return: self.__wavelengths.
        :rtype: list
        """

        return self.x.wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.__wavelengths** private attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "wavelengths"))

    @property
    def values(self):
        """
        Property for **self.__values** private attribute.

        :return: self.__values.
        :rtype: list
        """

        return np.array([self.get(wavelength)
                         for wavelength in self.wavelengths])

    @values.setter
    def values(self, value):
        """
        Setter for **self.__values** private attribute.

        :param value: Attribute value.
        :type value: list
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "values"))

    @property
    def shape(self):
        """
        Property for **self.__shape** private attribute.

        :return: self.__shape.
        :rtype: tuple
        """

        return self.x.shape

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.__shape** private attribute.

        :param value: Attribute value.
        :type value: tuple
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(
            self.__class__.__name__, "shape"))

    def __hash__(self):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :return: Object hash.
        :rtype: int
        """

        return hash(id(self))

    def __getitem__(self, wavelength):
        """
        Reimplements the :meth:`object.__getitem__` method.

        :param wavelength: Wavelength.
        :type wavelength: float
        :return: Value.
        :rtype: ndarray
        """

        return np.array(
            (self.x[wavelength], self.y[wavelength], self.z[wavelength]))

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

    def __eq__(self, tri_spd):
        """
        Reimplements the :meth:`object.__eq__` method.

        :param tri_spd: Tri-spectral power distribution to compare for \
        equality.
        :type tri_spd: TriSpectralPowerDistribution
        :return: Tri-spectral power distribution equality.
        :rtype: bool
        """

        equality = True
        for axis in self.__mapping:
            equality *= getattr(self, axis) == getattr(tri_spd, axis)

        return equality

    def __ne__(self, tri_spd):
        """
        Reimplements the :meth:`object.__eq__` method.

        :param tri_spd: Tri-spectral power distribution to compare for \
        inequality.
        :type tri_spd: TriSpectralPowerDistribution
        :return: Tri-spectral power distribution inequality.
        :rtype: bool
        """

        return not (self == tri_spd)

    def __add__(self, x):
        """
        Reimplements the :meth:`object.__add__` method.

        :param x: Variable to add.
        :type x: float or array_like
        :return: Variable added tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        values = self.values + x

        for i, axis in enumerate(("x", "y", "z")):
            self.__data[axis].data = dict(
                tuple(zip(self.wavelengths, values[:, i])))

        return self

    def __sub__(self, x):
        """
        Reimplements the :meth:`object.__sub__` method.

        :param x: Variable to subtract.
        :type x: float or array_like
        :return: Variable subtracted tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        return self + (-x)

    def __mul__(self, x):
        """
        Reimplements the :meth:`object.__mul__` method.

        :param x: Variable to multiply.
        :type x: float or array_like
        :return: Variable multiplied tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        values = self.values * x

        for i, axis in enumerate(("x", "y", "z")):
            self.__data[axis].data = dict(
                tuple(zip(self.wavelengths, values[:, i])))

        return self

    def __div__(self, x):
        """
        Reimplements the :meth:`object.__div__` method.

        :param x: Variable to divide.
        :type x: float or array_like
        :return: Variable divided tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        return self * (1. / x)

    # Python 3 compatibility.
    __truediv__ = __div__

    def is_uniform(self):
        """
        Returns if the tri-spectral power distribution have uniformly spaced
        data.

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
        Extrapolates the tri-spectral power distribution according to
        *CIE 15:2004* recommendation.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :return: Extrapolated tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution

        References:

        -  `CIE 015:2004 Colorimetry, 3rd edition: \
        7.2.2.1 Extrapolation \
        <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_
        -  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: \
        10. EXTRAPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        """

        for i in self.__mapping.keys():
            getattr(self, i).extrapolate(start, end)

        return self

    def interpolate(self, start=None, end=None, steps=None):
        """
        Interpolates the tri-spectral power distribution following
        *CIE 167:2005* recommendations.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Interpolated tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution

        References:

        -  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: \
        9. INTERPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)

        return self

    def align(self, start, end, steps):
        """
        Aligns the tri-spectral power distribution to given shape: Interpolates
        first then extrapolates to fit the given range.

        :param start: Wavelengths range start in nm.
        :type start: float
        :param end: Wavelengths range end in nm.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Aligned tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(start, end, steps)
            getattr(self, i).extrapolate(start, end)

        return self

    def zeros(self, start=None, end=None, steps=None):
        """
        Zeros fills the tri-spectral power distribution: Missing values will be
        replaced with zeros to fit the defined range.

        :param start: Wavelengths range start.
        :type start: float
        :param end: Wavelengths range end.
        :type end: float
        :param steps: Wavelengths range steps.
        :type steps: float
        :return: Zeros filled tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        for i in self.__mapping.keys():
            getattr(self, i).zeros(start, end, steps)

        return self

    def normalise(self, factor=1.):
        """
        Normalises the tri-spectral power distribution with given normalization
        factor.

        :param factor: Normalization factor
        :type factor: float
        :return: Normalised tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        maximum = max(np.ravel(self.values))
        for i in self.__mapping.keys():
            getattr(self, i) * (1. / maximum) * factor

        return self

    def clone(self):
        """
        Clones the tri-spectral power distribution.

        :return: Cloned tri-spectral power distribution.
        :rtype: TriSpectralPowerDistribution
        """

        return copy.deepcopy(self)