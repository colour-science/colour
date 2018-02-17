# -*- coding: utf-8 -*-
"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data:

-   :func:`colour.colorimetry.tristimulus_weighting_factors_ASTME202211`
-   :func:`colour.colorimetry.spectral_to_XYZ_integration`
-   :func:`colour.colorimetry.\
spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815`
-   :func:`colour.colorimetry.spectral_to_XYZ_ASTME30815`
-   :func:`colour.spectral_to_XYZ`
-   :func:`colour.wavelength_to_XYZ`

The default implementation is based on practise *ASTM E308-15* method.

References
----------
-   :cite:`ASTMInternational2011a` : ASTM International. (2011). ASTM E2022-11
    - Standard Practice for Calculation of Weighting Factors for Tristimulus
    Integration. doi:10.1520/E2022-11
-   :cite:`ASTMInternational2015b` : ASTM International. (2015). ASTM E308-15 -
    Standard Practice for Computing the Colors of Objects by Using the CIE
    System. doi:10.1520/E0308-15
-   :cite:`Wyszecki2000bf` : Wyszecki, G., & Stiles, W. S. (2000). Integration
    Replaced by Summation. In Color Science: Concepts and Methods, Quantitative
    Data and Formulae (pp. 158-163). Wiley. ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import lagrange_coefficients
from colour.colorimetry import (DEFAULT_SPECTRAL_SHAPE, SpectralShape,
                                STANDARD_OBSERVERS_CMFS, ones_spd)
from colour.utilities import (CaseInsensitiveMapping, filter_kwargs, tsplit,
                              warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ASTME30815_PRACTISE_SHAPE', 'lagrange_coefficients_ASTME202211',
    'tristimulus_weighting_factors_ASTME202211',
    'adjust_tristimulus_weighting_factors_ASTME30815',
    'spectral_to_XYZ_integration',
    'spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815',
    'spectral_to_XYZ_ASTME30815', 'SPECTRAL_TO_XYZ_METHODS', 'spectral_to_XYZ',
    'wavelength_to_XYZ'
]

ASTME30815_PRACTISE_SHAPE = DEFAULT_SPECTRAL_SHAPE
ASTME30815_PRACTISE_SHAPE.__doc__ = """
Shape for *ASTM E308-15* practise: (360, 780, 1).

References
----------
-   :cite:`ASTMInternational2015b`

ASTME30815_PRACTISE_SHAPE : SpectralShape
"""

_LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE = None

_TRISTIMULUS_WEIGHTING_FACTORS_CACHE = None


def lagrange_coefficients_ASTME202211(interval=10, interval_type='inner'):
    """
    Computes the *Lagrange Coefficients* for given interval size using practise
    *ASTM E2022-11* method.

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

    References
    ----------
    -   :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> lagrange_coefficients_ASTME202211(10, 'inner')
    ... # doctest: +ELLIPSIS
    array([[-0.028...,  0.940...,  0.104..., -0.016...],
           [-0.048...,  0.864...,  0.216..., -0.032...],
           [-0.059...,  0.773...,  0.331..., -0.045...],
           [-0.064...,  0.672...,  0.448..., -0.056...],
           [-0.062...,  0.562...,  0.562..., -0.062...],
           [-0.056...,  0.448...,  0.672..., -0.064...],
           [-0.045...,  0.331...,  0.773..., -0.059...],
           [-0.032...,  0.216...,  0.864..., -0.048...],
           [-0.016...,  0.104...,  0.940..., -0.028...]])
    >>> lagrange_coefficients_ASTME202211(10, 'boundary')
    ... # doctest: +ELLIPSIS
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

    lica = _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE[name_lica] = (np.asarray(
        [lagrange_coefficients(r, d) for r in r_n]))

    return lica


def tristimulus_weighting_factors_ASTME202211(cmfs, illuminant, shape):
    """
    Returns a table of tristimulus weighting factors for given colour matching
    functions and illuminant using practise *ASTM E2022-11* method.

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
        :attr:`colour.colorimetry.tristimulus.\
_TRISTIMULUS_WEIGHTING_FACTORS_CACHE` attribute. Their identifier key is
        defined by the colour matching functions and illuminant names along
        the current shape such as:
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

    References
    ----------
    -   :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> from colour import (CMFS, CIE_standard_illuminant_A_function,
    ...     SpectralPowerDistribution, SpectralShape, numpy_print_options)
    >>> cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
    >>> wl = cmfs.shape.range()
    >>> A = SpectralPowerDistribution(
    ...     dict(zip(wl, CIE_standard_illuminant_A_function(wl))),
    ...     name='A (360, 830, 1)')
    >>> with numpy_print_options(suppress=True):
    ...     tristimulus_weighting_factors_ASTME202211(
    ...         cmfs, A, SpectralShape(360, 830, 20))
    ... # doctest: +ELLIPSIS
    array([[ -0.0002981...,  -0.0000317...,  -0.0013301...],
           [ -0.0087155...,  -0.0008915...,  -0.0407436...],
           [  0.0599679...,   0.0050203...,   0.2565018...],
           [  0.7734225...,   0.0779839...,   3.6965732...],
           [  1.9000905...,   0.3037005...,   9.7554195...],
           [  1.9707727...,   0.8552809...,  11.4867325...],
           [  0.7183623...,   2.1457000...,   6.7845806...],
           [  0.0426667...,   4.8985328...,   2.3208000...],
           [  1.5223302...,   9.6471138...,   0.7430671...],
           [  5.6770329...,  14.4609708...,   0.1958194...],
           [ 12.4451744...,  17.4742541...,   0.0051827...],
           [ 20.5535772...,  17.5838219...,  -0.0026512...],
           [ 25.3315384...,  14.8957035...,   0.       ...],
           [ 21.5711570...,  10.0796619...,   0.       ...],
           [ 12.1785817...,   5.0680655...,   0.       ...],
           [  4.6675746...,   1.8303239...,   0.       ...],
           [  1.3236117...,   0.5129694...,   0.       ...],
           [  0.3175325...,   0.1230084...,   0.       ...],
           [  0.0746341...,   0.0290243...,   0.       ...],
           [  0.0182990...,   0.0071606...,   0.       ...],
           [  0.0047942...,   0.0018888...,   0.       ...],
           [  0.0013293...,   0.0005277...,   0.       ...],
           [  0.0004254...,   0.0001704...,   0.       ...],
           [  0.0000962...,   0.0000389...,   0.       ...]])
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

    interval_i = np.int_(shape.interval)
    W = S[::interval_i, np.newaxis] * Y[::interval_i, :]

    # First and last measurement intervals *Lagrange Coefficients*.
    c_c = lagrange_coefficients_ASTME202211(interval_i, 'boundary')
    # Intermediate measurement intervals *Lagrange Coefficients*.
    c_b = lagrange_coefficients_ASTME202211(interval_i, 'inner')

    # Total wavelengths count.
    w_c = len(Y)
    # Measurement interval interpolated values count.
    r_c = c_b.shape[0]
    # Last interval first interpolated wavelength.
    w_lif = w_c - (w_c - 1) % interval_i - 1 - r_c

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
                W[k, i] = (W[k, i] + c_c[r_c - j - 1, i_cm - k] * S[j + w_lif]
                           * Y[j + w_lif, i])

        # Intermediate intervals.
        for j in range(i_c - 3):
            for k in range(r_c):
                w_i = (r_c + 1) * (j + 1) + 1 + k
                W[j, i] = W[j, i] + c_b[k, 0] * S[w_i] * Y[w_i, i]
                W[j + 1, i] = W[j + 1, i] + c_b[k, 1] * S[w_i] * Y[w_i, i]
                W[j + 2, i] = W[j + 2, i] + c_b[k, 2] * S[w_i] * Y[w_i, i]
                W[j + 3, i] = W[j + 3, i] + c_b[k, 3] * S[w_i] * Y[w_i, i]

        # Extrapolation of potential incomplete interval.
        for j in range(int(w_c - ((w_c - 1) % interval_i)), w_c, 1):
            W[i_cm, i] = W[i_cm, i] + S[j] * Y[j, i]

    W *= 100 / np.sum(W, axis=0)[1]

    _TRISTIMULUS_WEIGHTING_FACTORS_CACHE[name_twf] = W

    return W


def adjust_tristimulus_weighting_factors_ASTME30815(W, shape_r, shape_t):
    """
    Adjusts given table of tristimulus weighting factors to account for a
    shorter wavelengths range of the test spectral shape compared to the
    reference spectral shape using practise  *ASTM E308-15* method:
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

    References
    ----------
    -   :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import (CMFS, CIE_standard_illuminant_A_function,
    ...     SpectralPowerDistribution, SpectralShape)
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
    >>> wl = cmfs.shape.range()
    >>> A = SpectralPowerDistribution(
    ...     dict(zip(wl, CIE_standard_illuminant_A_function(wl))),
    ...     name='A (360, 830, 1)')
    >>> W = tristimulus_weighting_factors_ASTME202211(
    ...     cmfs, A, SpectralShape(360, 830, 20))
    >>> with numpy_print_options(suppress=True):
    ...     adjust_tristimulus_weighting_factors_ASTME30815(
    ...         W,  SpectralShape(360, 830, 20), SpectralShape(400, 700, 20))
    ... # doctest: +ELLIPSIS
    array([[  0.0509543...,   0.0040971...,   0.2144280...],
           [  0.7734225...,   0.0779839...,   3.6965732...],
           [  1.9000905...,   0.3037005...,   9.7554195...],
           [  1.9707727...,   0.8552809...,  11.4867325...],
           [  0.7183623...,   2.1457000...,   6.7845806...],
           [  0.0426667...,   4.8985328...,   2.3208000...],
           [  1.5223302...,   9.6471138...,   0.7430671...],
           [  5.6770329...,  14.4609708...,   0.1958194...],
           [ 12.4451744...,  17.4742541...,   0.0051827...],
           [ 20.5535772...,  17.5838219...,  -0.0026512...],
           [ 25.3315384...,  14.8957035...,   0.       ...],
           [ 21.5711570...,  10.0796619...,   0.       ...],
           [ 12.1785817...,   5.0680655...,   0.       ...],
           [  4.6675746...,   1.8303239...,   0.       ...],
           [  1.3236117...,   0.5129694...,   0.       ...],
           [  0.4171109...,   0.1618194...,   0.       ...]])
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
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=ones_spd(STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer'].shape)):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant according to
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
    -   :cite:`Wyszecki2000bf`

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
    ...     700: 0.2852
    ... }
    >>> spd = SpectralPowerDistribution(data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_integration(spd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 11.5296285...,   9.9499467...,   4.7066079...])
    """

    if illuminant.shape != cmfs.shape:
        warning('Aligning "{0}" illuminant shape to "{1}" colour matching '
                'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    if spd.shape != cmfs.shape:
        warning('Aligning "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(spd.name, cmfs.name))
        spd = spd.copy().align(cmfs.shape)

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
    of tristimulus weighting factors according to practise
    *ASTM E308-15* method.

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
    -   :cite:`ASTMInternational2015b`

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
    ...     700: 0.2852
    ... }
    >>> spd = SpectralPowerDistribution(data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815(
    ...     spd, cmfs, illuminant)  # doctest: +ELLIPSIS
    array([ 11.5296311...,   9.9505845...,   4.7098037...])
    """

    if illuminant.shape != cmfs.shape:
        warning('Aligning "{0}" illuminant shape to "{1}" colour matching '
                'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    if spd.shape.boundaries != cmfs.shape.boundaries:
        warning('Trimming "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(
                    illuminant.name, cmfs.name))
        spd = spd.copy().trim(cmfs.shape)

    W = tristimulus_weighting_factors_ASTME202211(
        cmfs, illuminant,
        SpectralShape(cmfs.shape.start, cmfs.shape.end, spd.shape.interval))
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
    using given colour matching functions and illuminant according to
    practise *ASTM E308-15* method.

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
        if *True* this argument will trim the colour matching functions
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
        :attr:`colour.colorimetry.tristimulus.\
_TRISTIMULUS_WEIGHTING_FACTORS_CACHE` attribute. Their identifier key is
        defined by the colour matching functions and illuminant names along
        the current shape such as:
        `CIE 1964 10 Degree Standard Observer, A, (360.0, 830.0, 10.0)`
        Considering the above, one should be mindful that using similar colour
        matching functions and illuminant names but with different spectral
        data will lead to unexpected behaviour.
    -   The output range of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    References
    ----------
    -   :cite:`ASTMInternational2015b`

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
    ...     700: 0.2852
    ... }
    >>> spd = SpectralPowerDistribution(data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ_ASTME30815(spd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 11.5290265...,   9.9502091...,   4.7098882...])
    """

    if spd.shape.interval not in (1, 5, 10, 20):
        raise ValueError(
            'Tristimulus values conversion from spectral data according to '
            'practise "ASTM E308-15" should be performed on spectral data '
            'with measurement interval of 1, 5, 10 or 20nm!')

    if use_practice_range:
        cmfs = cmfs.copy().trim(ASTME30815_PRACTISE_SHAPE)

    method = spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815
    if spd.shape.interval == 1:
        method = spectral_to_XYZ_integration
    elif spd.shape.interval == 5 and mi_5nm_omission_method:
        if cmfs.shape.interval != 5:
            cmfs = cmfs.copy().interpolate(SpectralShape(interval=5))
        method = spectral_to_XYZ_integration
    elif spd.shape.interval == 20 and mi_20nm_interpolation_method:
        spd = spd.copy()
        if spd.shape.boundaries != cmfs.shape.boundaries:
            warning(
                'Trimming "{0}" spectral power distribution shape to "{1}" '
                'colour matching functions shape.'.format(
                    illuminant.name, cmfs.name))
            spd.trim(cmfs.shape)

        # Extrapolation of additional 20nm padding intervals.
        spd.align(SpectralShape(spd.shape.start - 20, spd.shape.end + 20, 10))
        for i in range(2):
            spd[spd.wavelengths[i]] = (
                3 * spd.values[i + 2] -
                3 * spd.values[i + 4] + spd.values[i + 6])  # yapf: disable
            i_e = len(spd.domain) - 1 - i
            spd[spd.wavelengths[i_e]] = (
                spd.values[i_e - 6] - 3 * spd.values[i_e - 4] +
                3 * spd.values[i_e - 2])

        # Interpolating every odd numbered values.
        # TODO: Investigate code vectorisation.
        for i in range(3, len(spd.domain) - 3, 2):
            spd[spd.wavelengths[i]] = (
                -0.0625 * spd.values[i - 3] + 0.5625 * spd.values[i - 1] +
                0.5625 * spd.values[i + 1] - 0.0625 * spd.values[i + 3])

        # Discarding the additional 20nm padding intervals.
        spd.trim(SpectralShape(spd.shape.start + 20, spd.shape.end - 20, 10))

    XYZ = method(spd, cmfs, illuminant)

    return XYZ


SPECTRAL_TO_XYZ_METHODS = CaseInsensitiveMapping({
    'ASTM E308-15': spectral_to_XYZ_ASTME30815,
    'Integration': spectral_to_XYZ_integration
})
SPECTRAL_TO_XYZ_METHODS.__doc__ = """
Supported spectral power distribution to *CIE XYZ* tristimulus values
conversion methods

References
----------
-   :cite:`ASTMInternational2011a`
-   :cite:`ASTMInternational2015b`
-   :cite:`Wyszecki2000bf`

SPECTRAL_TO_XYZ_METHODS : CaseInsensitiveMapping
    **{'ASTM E308-15', 'Integration'}**

Aliases:

-   'astm2015': 'ASTM E308-15'
"""
SPECTRAL_TO_XYZ_METHODS['astm2015'] = (SPECTRAL_TO_XYZ_METHODS['ASTM E308-15'])


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
        {:func:`colour.colorimetry.spectral_to_XYZ_ASTME30815`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        {:func:`colour.colorimetry.spectral_to_XYZ_ASTME30815`},
        5 nm measurement intervals spectral power distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        {:func:`colour.colorimetry.spectral_to_XYZ_ASTME30815`},
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

    References
    ----------
    -   :cite:`ASTMInternational2011a`
    -   :cite:`ASTMInternational2015b`
    -   :cite:`Wyszecki2000bf`

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
    ...     700: 0.2852
    ... }
    >>> spd = SpectralPowerDistribution(data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS['D50']
    >>> spectral_to_XYZ(spd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 11.5290265...,   9.9502091...,   4.7098882...])
    >>> spectral_to_XYZ(spd, cmfs, illuminant, use_practice_range=False)
    ... # doctest: +ELLIPSIS
    array([ 11.5291275...,   9.9502369...,   4.7098811...])
    >>> spectral_to_XYZ(spd, cmfs, illuminant, method='Integration')
    ... # doctest: +ELLIPSIS
    array([ 11.5296285...,   9.9499467...,   4.7066079...])
    """

    function = SPECTRAL_TO_XYZ_METHODS[method]

    return function(spd, cmfs, illuminant, **filter_kwargs(function, **kwargs))


def wavelength_to_XYZ(
        wavelength,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Converts given wavelength :math:`\lambda` to *CIE XYZ* tristimulus values
    using given colour matching functions.

    If the wavelength :math:`\lambda` is not available in the colour matching
    function, its value will be calculated according to *CIE 15:2004*
    recommendation: the method developed by *Sprague (1880)* will be used for
    interpolating functions having a uniformly spaced independent variable and
    the *Cubic Spline* method for non-uniformly spaced independent variable.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If wavelength :math:`\lambda` is not contained in the colour matching
        functions domain.

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in range [0, 1].

    Examples
    --------
    >>> from colour import CMFS
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> wavelength_to_XYZ(480, cmfs)  # doctest: +ELLIPSIS
    array([ 0.09564  ,  0.13902  ,  0.812950...])
    >>> wavelength_to_XYZ(480.5, cmfs)  # doctest: +ELLIPSIS
    array([ 0.0914287...,  0.1418350...,  0.7915726...])
    """

    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
            np.max(wavelength) > cmfs_shape.end):
        raise ValueError('"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.
                         format(wavelength, cmfs_shape.start, cmfs_shape.end))

    XYZ = np.reshape(cmfs[np.ravel(wavelength)],
                     np.asarray(wavelength).shape + (3, ))

    return XYZ
