#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data:

-   :func:`tristimulus_weighting_factors_ASTME202211`
-   :func:`spectral_to_XYZ_integration`
-   :func:`spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815`
-   :func:`spectral_to_XYZ_ASTME30815`
-   :func:`spectral_to_XYZ`
-   :func:`wavelength_to_XYZ`

The default implementation is based on practise *ASTM E308-15* method [2]_.

References
----------
.. [1]  ASTM International. (2011). ASTM E2022–11 - Standard Practice for
        Calculation of Weighting Factors for Tristimulus Integration, i, 1–10.
        doi:10.1520/E2022-11
.. [2]  ASTM International. (2015). ASTM E308–15 - Standard Practice for
        Computing the Colors of Objects by Using the CIE System, 1–47.
        doi:10.1520/E0308-15

See Also
--------
`Colour Matching Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
`Spectrum Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/spectrum.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import (
    CubicSplineInterpolator,
    LinearInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    lagrange_coefficients)
from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE,
    SpectralShape,
    STANDARD_OBSERVERS_CMFS, ones_spd)
from colour.utilities import (
    CaseInsensitiveMapping,
    filter_kwargs,
    is_string,
    tsplit,
    warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ASTME30815_PRACTISE_SHAPE',
           'lagrange_coefficients_ASTME202211',
           'tristimulus_weighting_factors_ASTME202211',
           'adjust_tristimulus_weighting_factors_ASTME30815',
           'spectral_to_XYZ_integration',
           'spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815',
           'spectral_to_XYZ_ASTME30815',
           'SPECTRAL_TO_XYZ_METHODS',
           'spectral_to_XYZ',
           'wavelength_to_XYZ']

ASTME30815_PRACTISE_SHAPE = DEFAULT_SPECTRAL_SHAPE
"""
*ASTM E308–15* practise shape: (360, 780, 1).

ASTME30815_PRACTISE_SHAPE : SpectralShape
"""

_LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE = None

_TRISTIMULUS_WEIGHTING_FACTORS_CACHE = None


def lagrange_coefficients_ASTME202211(
        interval=10,
        interval_type='inner'):
    """
    Computes the *Lagrange Coefficients* for given interval size using practise
    *ASTM E2022-11* method [1]_.

    Parameters
    ----------
    interval : int
        Interval size in nm.
    interval_type : unicode, optional
        **{'inner', 'boundary'}**,
        If the interval is an *inner* interval *Lagrange Coefficients* are
        computed for degree 4. Degree 3 is used for a *boundary* interval.

    Returns
    -------
    ndarray
        *Lagrange Coefficients*.

    See Also
    --------
    colour.lagrange_coefficients

    Examples
    --------
    >>> lagrange_coefficients_ASTME202211(  # doctest: +ELLIPSIS
    ...     10, 'inner')
    array([[-0.028...,  0.940...,  0.104..., -0.016...],
           [-0.048...,  0.864...,  0.216..., -0.032...],
           [-0.059...,  0.773...,  0.331..., -0.045...],
           [-0.064...,  0.672...,  0.448..., -0.056...],
           [-0.062...,  0.562...,  0.562..., -0.062...],
           [-0.056...,  0.448...,  0.672..., -0.064...],
           [-0.045...,  0.331...,  0.773..., -0.059...],
           [-0.032...,  0.216...,  0.864..., -0.048...],
           [-0.016...,  0.104...,  0.940..., -0.028...]])
    >>> lagrange_coefficients_ASTME202211(  # doctest: +ELLIPSIS
    ...     10, 'boundary')
    array([[ 0.85...,  0.19..., -0.04...],
           [ 0.72...,  0.36..., -0.08...],
           [ 0.59...,  0.51..., -0.10...],
           [ 0.48...,  0.64..., -0.12...],
           [ 0.37...,  0.75..., -0.12...],
           [ 0.28...,  0.84..., -0.12...],
           [ 0.19...,  0.91..., -0.10...],
           [ 0.12...,  0.96..., -0.08...],
           [ 0.05...,  0.99..., -0.04...]])
    """

    global _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE
    if _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE is None:
        _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE = CaseInsensitiveMapping()

    name_lica = ', '.join((str(interval), interval_type))
    if name_lica in _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE:
        return _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE[name_lica]

    r_n = np.linspace(1 / interval, 1 - (1 / interval), interval - 1)
    d = 3
    if interval_type.lower() == 'inner':
        r_n += 1
        d = 4

    lica = _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE[name_lica] = (
        np.asarray([lagrange_coefficients(r, d) for r in r_n]))

    return lica


def tristimulus_weighting_factors_ASTME202211(cmfs, illuminant, shape):
    """
    Returns a table of tristimulus weighting factors for given colour matching
    functions and illuminant using practise *ASTM E2022-11* method [1]_.

    The computed table of tristimulus weighting factors should be used with
    spectral data that has been corrected for spectral bandpass dependence.

    Parameters
    ----------
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution
        Illuminant spectral power distribution.
    shape : SpectralShape
        Shape used to build the table, only the interval is needed.

    Returns
    -------
    ndarray
        Tristimulus weighting factors table.

    Raises
    ------
    ValueError
        If the colour matching functions or illuminant intervals are not equal
        to 1 nm.

    Warning
    -------
    -   The tables of tristimulus weighting factors are cached in
        :attr:`_TRISTIMULUS_WEIGHTING_FACTORS_CACHE` attribute. Their
        identifier key is defined by the colour matching functions and
        illuminant names along the current shape such as:
        `CIE 1964 10 Degree Standard Observer, A, (360.0, 830.0, 10.0)`
        Considering the above, one should be mindful that using similar colour
        matching functions and illuminant names but with different spectral
        data will lead to unexpected behaviour.

    Notes
    -----
    -   Input colour matching functions and illuminant intervals are expected
        to be equal to 1 nm. If the illuminant data is not available at 1 nm
        interval, it needs to be interpolated using *CIE* recommendations:
        The method developed by *Sprague (1880)* should be used for
        interpolating functions having a uniformly spaced independent variable
        and a *Cubic Spline* method for non-uniformly spaced independent
        variable.

    Examples
    --------
    >>> from colour import (
    ...     CMFS,
    ...     CIE_standard_illuminant_A_function,
    ...     SpectralPowerDistribution,
    ...     SpectralShape)
    >>> cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
    >>> wl = cmfs.shape.range()
    >>> A = SpectralPowerDistribution(
    ...     'A (360, 830, 1)',
    ...     dict(zip(wl, CIE_standard_illuminant_A_function(wl))))
    >>> tristimulus_weighting_factors_ASTME202211(  # doctest: +ELLIPSIS
    ...     cmfs, A, SpectralShape(360, 830, 20))
    array([[ -2.9816934...e-04,  -3.1709762...e-05,  -1.3301218...e-03],
           [ -8.7154955...e-03,  -8.9154168...e-04,  -4.0743684...e-02],
           [  5.9967988...e-02,   5.0203497...e-03,   2.5650183...e-01],
           [  7.7342255...e-01,   7.7983983...e-02,   3.6965732...e+00],
           [  1.9000905...e+00,   3.0370051...e-01,   9.7554195...e+00],
           [  1.9707727...e+00,   8.5528092...e-01,   1.1486732...e+01],
           [  7.1836236...e-01,   2.1457000...e+00,   6.7845806...e+00],
           [  4.2666758...e-02,   4.8985328...e+00,   2.3208000...e+00],
           [  1.5223302...e+00,   9.6471138...e+00,   7.4306714...e-01],
           [  5.6770329...e+00,   1.4460970...e+01,   1.9581949...e-01],
           [  1.2445174...e+01,   1.7474254...e+01,   5.1826979...e-03],
           [  2.0553577...e+01,   1.7583821...e+01,  -2.6512696...e-03],
           [  2.5331538...e+01,   1.4895703...e+01,   0.0000000...e+00],
           [  2.1571157...e+01,   1.0079661...e+01,   0.0000000...e+00],
           [  1.2178581...e+01,   5.0680655...e+00,   0.0000000...e+00],
           [  4.6675746...e+00,   1.8303239...e+00,   0.0000000...e+00],
           [  1.3236117...e+00,   5.1296946...e-01,   0.0000000...e+00],
           [  3.1753258...e-01,   1.2300847...e-01,   0.0000000...e+00],
           [  7.4634128...e-02,   2.9024389...e-02,   0.0000000...e+00],
           [  1.8299016...e-02,   7.1606335...e-03,   0.0000000...e+00],
           [  4.7942065...e-03,   1.8888730...e-03,   0.0000000...e+00],
           [  1.3293045...e-03,   5.2774591...e-04,   0.0000000...e+00],
           [  4.2546928...e-04,   1.7041978...e-04,   0.0000000...e+00],
           [  9.6251115...e-05,   3.8955295...e-05,   0.0000000...e+00]])
    """

    if cmfs.shape.interval != 1:
        raise ValueError('"{0}" shape "interval" must be 1!'.format(cmfs))

    if illuminant.shape.interval != 1:
        raise ValueError(
            '"{0}" shape "interval" must be 1!'.format(illuminant))

    global _TRISTIMULUS_WEIGHTING_FACTORS_CACHE
    if _TRISTIMULUS_WEIGHTING_FACTORS_CACHE is None:
        _TRISTIMULUS_WEIGHTING_FACTORS_CACHE = CaseInsensitiveMapping()

    name_twf = ', '.join((cmfs.name, illuminant.name, str(shape)))
    if name_twf in _TRISTIMULUS_WEIGHTING_FACTORS_CACHE:
        return _TRISTIMULUS_WEIGHTING_FACTORS_CACHE[name_twf]

    Y = cmfs.values
    S = illuminant.values

    W = S[::shape.interval, np.newaxis] * Y[::shape.interval, :]

    # First and last measurement intervals *Lagrange Coefficients*.
    c_c = lagrange_coefficients_ASTME202211(shape.interval, 'boundary')
    # Intermediate measurement intervals *Lagrange Coefficients*.
    c_b = lagrange_coefficients_ASTME202211(shape.interval, 'inner')

    # Total wavelengths count.
    w_c = len(Y)
    # Measurement interval interpolated values count.
    r_c = c_b.shape[0]
    # Last interval first interpolated wavelength.
    w_lif = w_c - (w_c - 1) % shape.interval - 1 - r_c

    # Intervals count.
    i_c = W.shape[0]
    i_cm = i_c - 1

    for i in range(3):
        # First interval.
        for j in range(r_c):
            for k in range(3):
                W[k, i] = W[k, i] + c_c[j, k] * S[j + 1] * Y[j + 1, i]

        # Last interval.
        for j in range(r_c):
            for k in range(i_cm, i_cm - 3, -1):
                W[k, i] = (W[k, i] + c_c[r_c - j - 1, i_cm - k] *
                           S[j + w_lif] * Y[j + w_lif, i])

        # Intermediate intervals.
        for j in range(i_c - 3):
            for k in range(r_c):
                w_i = (r_c + 1) * (j + 1) + 1 + k
                W[j, i] = W[j, i] + c_b[k, 0] * S[w_i] * Y[w_i, i]
                W[j + 1, i] = W[j + 1, i] + c_b[k, 1] * S[w_i] * Y[w_i, i]
                W[j + 2, i] = W[j + 2, i] + c_b[k, 2] * S[w_i] * Y[w_i, i]
                W[j + 3, i] = W[j + 3, i] + c_b[k, 3] * S[w_i] * Y[w_i, i]

        # Extrapolation of potential incomplete interval.
        for j in range(int(w_c - ((w_c - 1) % shape.interval)), w_c, 1):
            W[i_cm, i] = W[i_cm, i] + S[j] * Y[j, i]

    W *= 100 / np.sum(W, axis=0)[1]

    _TRISTIMULUS_WEIGHTING_FACTORS_CACHE[name_twf] = W

    return W


def adjust_tristimulus_weighting_factors_ASTME30815(W, shape_r, shape_t):
    """
    Adjusts given table of tristimulus weighting factors to account for a
    shorter wavelengths range of the test spectral shape compared to the
    reference spectral shape using practise  *ASTM E308-15* method [2]_:
    Weights at the wavelengths for which data are not available are added to
    the weights at the shortest and longest wavelength for which spectral data
    are available.

    Parameters
    ----------
    W : array_like
        Tristimulus weighting factors table.
    shape_r : SpectralShape
        Reference spectral shape.
    shape_t : SpectralShape
        Test spectral shape.

    Returns
    -------
    ndarray
        Adjusted tristimulus weighting factors.

    Examples
    --------
    >>> from colour import (
    ...     CMFS,
    ...     CIE_standard_illuminant_A_function,
    ...     SpectralPowerDistribution,
    ...     SpectralShape)
    >>> cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
    >>> wl = cmfs.shape.range()
    >>> A = SpectralPowerDistribution(
    ...     'A (360, 830, 1)',
    ...     dict(zip(wl, CIE_standard_illuminant_A_function(wl))))
    >>> W = tristimulus_weighting_factors_ASTME202211(
    ...     cmfs, A, SpectralShape(360, 830, 20))
    >>> adjust_tristimulus_weighting_factors_ASTME30815(  # doctest: +ELLIPSIS
    ...     W, SpectralShape(360, 830, 20), SpectralShape(400, 700, 20))
    array([[  5.0954324...e-02,   4.0970982...e-03,   2.1442802...e-01],
           [  7.7342255...e-01,   7.7983983...e-02,   3.6965732...e+00],
           [  1.9000905...e+00,   3.0370051...e-01,   9.7554195...e+00],
           [  1.9707727...e+00,   8.5528092...e-01,   1.1486732...e+01],
           [  7.1836236...e-01,   2.1457000...e+00,   6.7845806...e+00],
           [  4.2666758...e-02,   4.8985328...e+00,   2.3208000...e+00],
           [  1.5223302...e+00,   9.6471138...e+00,   7.4306714...e-01],
           [  5.6770329...e+00,   1.4460970...e+01,   1.9581949...e-01],
           [  1.2445174...e+01,   1.7474254...e+01,   5.1826979...e-03],
           [  2.0553577...e+01,   1.7583821...e+01,  -2.6512696...e-03],
           [  2.5331538...e+01,   1.4895703...e+01,   0.0000000...e+00],
           [  2.1571157...e+01,   1.0079661...e+01,   0.0000000...e+00],
           [  1.2178581...e+01,   5.0680655...e+00,   0.0000000...e+00],
           [  4.6675746...e+00,   1.8303239...e+00,   0.0000000...e+00],
           [  1.3236117...e+00,   5.1296946...e-01,   0.0000000...e+00],
           [  4.1711096...e-01,   1.6181949...e-01,   0.0000000...e+00]])
    """

    W = np.copy(W)

    start_index = int((shape_t.start - shape_r.start) / shape_r.interval)
    for i in range(start_index):
        W[start_index] += W[i]

    end_index = int((shape_r.end - shape_t.end) / shape_r.interval)
    for i in range(end_index):
        W[-end_index - 1] += W[-i - 1]

    return W[start_index:-end_index or None, ...]


def spectral_to_XYZ_integration(
        spd,
        cmfs=STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer'],
        illuminant=ones_spd(
            STANDARD_OBSERVERS_CMFS[
                'CIE 1931 2 Degree Standard Observer'].shape)):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant accordingly to
    classical integration method.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution, optional
        Illuminant spectral power distribution.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The output range of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    References
    ----------
    .. [3]  Wyszecki, G., & Stiles, W. S. (2000). Integration Replace by
            Summation. In Color Science: Concepts and Methods, Quantitative
            Data and Formulae (pp. 158–163). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> from colour import (
    ...     CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution)
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> data = {
    ...     400: 0.0641,
    ...     420: 0.0645,
    ...     440: 0.0562,
    ...     460: 0.0537,
    ...     480: 0.0559,
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360,
    ...     620: 0.1511,
    ...     640: 0.1688,
    ...     660: 0.1996,
    ...     680: 0.2397,
    ...     700: 0.2852}
    >>> spd = SpectralPowerDistribution('Sample', data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_integration(  # doctest: +ELLIPSIS
    ...     spd, cmfs, illuminant)
    array([ 11.5296285...,   9.9499467...,   4.7066079...])
    """

    if illuminant.shape != cmfs.shape:
        warning('Aligning "{0}" illuminant shape to "{1}" colour matching '
                'functions shape.'.format(illuminant, cmfs))
        illuminant = illuminant.clone().align(cmfs.shape)

    if spd.shape != cmfs.shape:
        warning('Aligning "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(spd, cmfs))
        spd = spd.clone().align(cmfs.shape)

    S = illuminant.values
    x_bar, y_bar, z_bar = tsplit(cmfs.values)
    R = spd.values
    dw = cmfs.shape.interval

    k = 100 / (np.sum(y_bar * S) * dw)

    X_p = R * x_bar * S * dw
    Y_p = R * y_bar * S * dw
    Z_p = R * z_bar * S * dw

    XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)

    return XYZ


def spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815(
        spd,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=ones_spd(ASTME30815_PRACTISE_SHAPE)):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant using a table
    of tristimulus weighting factors accordingly to practise
    *ASTM E308-15* method [2]_.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution, optional
        Illuminant spectral power distribution.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The output range of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> from colour import (
    ...     CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution)
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> data = {
    ...     400: 0.0641,
    ...     420: 0.0645,
    ...     440: 0.0562,
    ...     460: 0.0537,
    ...     480: 0.0559,
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360,
    ...     620: 0.1511,
    ...     640: 0.1688,
    ...     660: 0.1996,
    ...     680: 0.2397,
    ...     700: 0.2852}
    >>> spd = SpectralPowerDistribution('Sample', data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815(
    ...     spd, cmfs, illuminant)  # doctest: +ELLIPSIS
    array([ 11.5296311...,   9.9505845...,   4.7098037...])
    """

    if illuminant.shape != cmfs.shape:
        warning('Aligning "{0}" illuminant shape to "{1}" colour matching '
                'functions shape.'.format(illuminant, cmfs))
        illuminant = illuminant.clone().align(cmfs.shape)

    if spd.shape.boundaries != cmfs.shape.boundaries:
        warning('Trimming "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(illuminant, cmfs))
        spd = spd.clone().trim_wavelengths(cmfs.shape)

    W = tristimulus_weighting_factors_ASTME202211(
        cmfs, illuminant, SpectralShape(
            cmfs.shape.start, cmfs.shape.end, spd.shape.interval))
    start_w = cmfs.shape.start
    end_w = cmfs.shape.start + spd.shape.interval * (W.shape[0] - 1)
    W = adjust_tristimulus_weighting_factors_ASTME30815(
        W, SpectralShape(start_w, end_w, spd.shape.interval), spd.shape)
    R = spd.values

    XYZ = np.sum(W * R[..., np.newaxis], axis=0)

    return XYZ


def spectral_to_XYZ_ASTME30815(
        spd,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=ones_spd(ASTME30815_PRACTISE_SHAPE),
        use_practice_range=True,
        mi_5nm_omission_method=True,
        mi_20nm_interpolation_method=True):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant accordingly to
    practise *ASTM E308-15* method [2]_.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution, optional
        Illuminant spectral power distribution.
    use_practice_range : bool, optional
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if `True` this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        5 nm measurement intervals spectral power distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        20 nm measurement intervals spectral power distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Warning
    -------
    -   The tables of tristimulus weighting factors are cached in
        :attr:`_TRISTIMULUS_WEIGHTING_FACTORS_CACHE` attribute. Their
        identifier key is defined by the colour matching functions and
        illuminant names along the current shape such as:
        `CIE 1964 10 Degree Standard Observer, A, (360.0, 830.0, 10.0)`
        Considering the above, one should be mindful that using similar colour
        matching functions and illuminant names but with different spectral
        data will lead to unexpected behaviour.
    -   The output range of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> from colour import (
    ...     CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution)
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> data = {
    ...     400: 0.0641,
    ...     420: 0.0645,
    ...     440: 0.0562,
    ...     460: 0.0537,
    ...     480: 0.0559,
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360,
    ...     620: 0.1511,
    ...     640: 0.1688,
    ...     660: 0.1996,
    ...     680: 0.2397,
    ...     700: 0.2852}
    >>> spd = SpectralPowerDistribution('Sample', data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_ASTME30815(
    ...     spd, cmfs, illuminant)  # doctest: +ELLIPSIS
    array([ 11.5290265...,   9.9502091...,   4.7098882...])
    """

    if spd.shape.interval not in (1, 5, 10, 20):
        raise ValueError(
            'Tristimulus values conversion from spectral data accordingly to '
            'practise "ASTM E308-15" should be performed on spectral data '
            'with measurement interval of 1, 5, 10 or 20nm!')

    if use_practice_range:
        cmfs = cmfs.clone().trim_wavelengths(ASTME30815_PRACTISE_SHAPE)

    method = spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815
    if spd.shape.interval == 1:
        method = spectral_to_XYZ_integration
    elif spd.shape.interval == 5 and mi_5nm_omission_method:
        if cmfs.shape.interval != 5:
            cmfs = cmfs.clone().interpolate(SpectralShape(interval=5))
        method = spectral_to_XYZ_integration
    elif spd.shape.interval == 20 and mi_20nm_interpolation_method:
        spd = spd.clone()
        if spd.shape.boundaries != cmfs.shape.boundaries:
            warning(
                'Trimming "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(illuminant, cmfs))
            spd.trim_wavelengths(cmfs.shape)

        # Extrapolation of additional 20nm padding intervals.
        spd.align(SpectralShape(spd.shape.start - 20, spd.shape.end + 20, 10))
        for i in range(2):
            spd[spd.wavelengths[i]] = (3 * spd.values[i + 2] -
                                       3 * spd.values[i + 4] +
                                       spd.values[i + 6])
            i_e = len(spd) - 1 - i
            spd[spd.wavelengths[i_e]] = (spd.values[i_e - 6] -
                                         3 * spd.values[i_e - 4] +
                                         3 * spd.values[i_e - 2])

        # Interpolating every odd numbered values.
        # TODO: Investigate code vectorisation.
        for i in range(3, len(spd) - 3, 2):
            spd[spd.wavelengths[i]] = (-0.0625 * spd.values[i - 3] +
                                       0.5625 * spd.values[i - 1] +
                                       0.5625 * spd.values[i + 1] -
                                       0.0625 * spd.values[i + 3])

        # Discarding the additional 20nm padding intervals.
        spd.trim_wavelengths(SpectralShape(spd.shape.start + 20,
                                           spd.shape.end - 20,
                                           10))

    XYZ = method(spd, cmfs, illuminant)

    return XYZ


SPECTRAL_TO_XYZ_METHODS = CaseInsensitiveMapping(
    {'ASTM E308-15': spectral_to_XYZ_ASTME30815,
     'Integration': spectral_to_XYZ_integration})
"""
Supported spectral power distribution to *CIE XYZ* tristimulus values
conversion methods

SPECTRAL_TO_XYZ_METHODS : CaseInsensitiveMapping
    **{'ASTM E308-15', 'Integration'}**

Aliases:

-   'astm2015': 'ASTM E308-15'
"""
SPECTRAL_TO_XYZ_METHODS['astm2015'] = (
    SPECTRAL_TO_XYZ_METHODS['ASTM E308-15'])


def spectral_to_XYZ(
        spd,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=ones_spd(ASTME30815_PRACTISE_SHAPE),
        method='ASTM E308-15',
        **kwargs):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions, illuminant and method.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution, optional
        Illuminant spectral power distribution.
    method : unicode, optional
        **{'ASTM E308-15', 'Integration'}**,
        Computation method.

    Other Parameters
    ----------------
    use_practice_range : bool, optional
        {:func:`spectral_to_XYZ_ASTME30815`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if `True` this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        {:func:`spectral_to_XYZ_ASTME30815`},
        5 nm measurement intervals spectral power distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        {:func:`spectral_to_XYZ_ASTME30815`},
        20 nm measurement intervals spectral power distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The output range of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> from colour import (
    ...     CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution)
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> data = {
    ...     400: 0.0641,
    ...     420: 0.0645,
    ...     440: 0.0562,
    ...     460: 0.0537,
    ...     480: 0.0559,
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360,
    ...     620: 0.1511,
    ...     640: 0.1688,
    ...     660: 0.1996,
    ...     680: 0.2397,
    ...     700: 0.2852}
    >>> spd = SpectralPowerDistribution('Sample', data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ(  # doctest: +ELLIPSIS
    ...     spd, cmfs, illuminant)
    array([ 11.5290265...,   9.9502091...,   4.7098882...])
    >>> spectral_to_XYZ(  # doctest: +ELLIPSIS
    ...     spd, cmfs, illuminant, use_practice_range=False)
    array([ 11.5291275...,   9.9502369...,   4.7098811...])
    >>> spectral_to_XYZ(  # doctest: +ELLIPSIS
    ...     spd, cmfs, illuminant, method='Integration')
    array([ 11.5296285...,   9.9499467...,   4.7066079...])
    """

    function = SPECTRAL_TO_XYZ_METHODS[method]

    filter_kwargs(function, **kwargs)

    return function(spd, cmfs, illuminant, **kwargs)


def wavelength_to_XYZ(wavelength,
                      cmfs=STANDARD_OBSERVERS_CMFS[
                          'CIE 1931 2 Degree Standard Observer'],
                      method=None):
    """
    Converts given wavelength :math:`\lambda` to *CIE XYZ* tristimulus values
    using given colour matching functions.

    If the wavelength :math:`\lambda` is not available in the colour matching
    function, its value will be calculated using *CIE* recommendations:
    The method developed by *Sprague (1880)* should be used for interpolating
    functions having a uniformly spaced independent variable and a
    *Cubic Spline* method for non-uniformly spaced independent variable.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    method : unicode, optional
        {None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'},
        Enforce given interpolation method.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    RuntimeError
        If *Sprague (1880)* interpolation method is forced with a
        non-uniformly spaced independent variable.
    ValueError
        If the interpolation method is not defined or if wavelength
        :math:`\lambda` is not contained in the colour matching functions
        domain.

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 1].
    -   If *scipy* is not unavailable the *Cubic Spline* method will fallback
        to legacy *Linear* interpolation.
    -   *Sprague (1880)* interpolator cannot be used for interpolating
        functions having a non-uniformly spaced independent variable.

    Warning
    -------
    -   If *scipy* is not unavailable the *Cubic Spline* method will fallback
        to legacy *Linear* interpolation.
    -   *Cubic Spline* interpolator requires at least 3 wavelengths
        :math:`\lambda_n` for interpolation.
    -   *Linear* interpolator requires at least 2 wavelengths :math:`\lambda_n`
        for interpolation.
    -   *Pchip* interpolator requires at least 2 wavelengths :math:`\lambda_n`
        for interpolation.
    -   *Sprague (1880)* interpolator requires at least 6 wavelengths
        :math:`\lambda_n` for interpolation.

    Examples
    --------
    Uniform data is using *Sprague (1880)* interpolation by default:

    >>> from colour import CMFS
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> wavelength_to_XYZ(480, cmfs)  # doctest: +ELLIPSIS
    array([ 0.09564  ,  0.13902  ,  0.812950...])
    >>> wavelength_to_XYZ(480.5, cmfs)  # doctest: +ELLIPSIS
    array([ 0.0914287...,  0.1418350...,  0.7915726...])

    Enforcing *Cubic Spline* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Cubic Spline')  # doctest: +ELLIPSIS
    array([ 0.0914288...,  0.1418351...,  0.7915729...])

    Enforcing *Linear* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Linear')  # doctest: +ELLIPSIS
    array([ 0.0914697...,  0.1418482...,  0.7917337...])

    Enforcing *Pchip* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Pchip')  # doctest: +ELLIPSIS
    array([ 0.0914280...,  0.1418341...,  0.7915711...])
    """

    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
            np.max(wavelength) > cmfs_shape.end):
        raise ValueError(
            '"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, cmfs_shape.start, cmfs_shape.end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values

        if is_string(method):
            method = method.lower()

        is_uniform = cmfs.is_uniform()

        if method is None:
            if is_uniform:
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator
        elif method == 'cubic spline':
            interpolator = CubicSplineInterpolator
        elif method == 'linear':
            interpolator = LinearInterpolator
        elif method == 'pchip':
            interpolator = PchipInterpolator
        elif method == 'sprague':
            if is_uniform:
                interpolator = SpragueInterpolator
            else:
                raise RuntimeError(
                    ('"Sprague" interpolator can only be used for '
                     'interpolating functions having a uniformly spaced '
                     'independent variable!'))
        else:
            raise ValueError(
                'Undefined "{0}" interpolator!'.format(method))

        interpolators = [interpolator(wavelengths, values[..., i])
                         for i in range(values.shape[-1])]

        XYZ = np.dstack([i(np.ravel(wavelength)) for i in interpolators])
    else:
        XYZ = cmfs[wavelength]

    XYZ = np.reshape(XYZ, np.asarray(wavelength).shape + (3,))

    return XYZ
