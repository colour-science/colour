# -*- coding: utf-8 -*-
"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data:

-   :attr:`colour.SPECTRAL_SHAPE_ASTME308`
-   :func:`colour.colorimetry.tristimulus_weighting_factors_ASTME2022`
-   :func:`colour.colorimetry.sd_to_XYZ_integration`
-   :func:`colour.colorimetry.\
sd_to_XYZ_tristimulus_weighting_factors_ASTME308`
-   :func:`colour.colorimetry.sd_to_XYZ_ASTME308`
-   :attr:`colour.SD_TO_XYZ_METHODS`
-   :func:`colour.sd_to_XYZ`
-   :func:`colour.colorimetry.msds_to_XYZ_integration`
-   :func:`colour.colorimetry.msds_to_XYZ_ASTME308`
-   :attr:`colour.MSDS_TO_XYZ_METHODS`
-   :func:`colour.msds_to_XYZ`
-   :func:`colour.wavelength_to_XYZ`

The default implementation is based on practise *ASTM E308-15* method.

References
----------
-   :cite:`ASTMInternational2011a` : ASTM International. (2011). ASTM E2022-11
    - Standard Practice for Calculation of Weighting Factors for Tristimulus
    Integration (pp. 1-10). doi:10.1520/E2022-11
-   :cite:`ASTMInternational2015b` : ASTM International. (2015). ASTM E308-15 -
    Standard Practice for Computing the Colors of Objects by Using the CIE
    System (pp. 1-47). doi:10.1520/E0308-15
-   :cite:`Wyszecki2000bf` : Wyszecki, Günther, & Stiles, W. S. (2000).
    Integration Replaced by Summation. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (pp. 158-163). Wiley. ISBN:978-0-471-39918-6
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import lagrange_coefficients
from colour.colorimetry import (SPECTRAL_SHAPE_DEFAULT,
                                MultiSpectralDistributions, SpectralShape,
                                MSDS_CMFS_STANDARD_OBSERVER, sd_ones)
from colour.constants import DEFAULT_INT_DTYPE
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              filter_kwargs, from_range_100,
                              get_domain_range_scale, runtime_warning, tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SPECTRAL_SHAPE_ASTME308', 'lagrange_coefficients_ASTME2022',
    'tristimulus_weighting_factors_ASTME2022',
    'adjust_tristimulus_weighting_factors_ASTME308', 'sd_to_XYZ_integration',
    'sd_to_XYZ_tristimulus_weighting_factors_ASTME308', 'sd_to_XYZ_ASTME308',
    'SD_TO_XYZ_METHODS', 'sd_to_XYZ', 'msds_to_XYZ_integration',
    'msds_to_XYZ_ASTME308', 'MSDS_TO_XYZ_METHODS', 'msds_to_XYZ',
    'wavelength_to_XYZ'
]

SPECTRAL_SHAPE_ASTME308 = SPECTRAL_SHAPE_DEFAULT
SPECTRAL_SHAPE_ASTME308.__doc__ = """
Shape for *ASTM E308-15* practise: (360, 780, 1).

References
----------
:cite:`ASTMInternational2015b`

SPECTRAL_SHAPE_ASTME308 : SpectralShape
"""

_CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS = None

_CACHE_TRISTIMULUS_WEIGHTING_FACTORS = None

_CACHE_SD_TO_XYZ = None


def lagrange_coefficients_ASTME2022(interval=10, interval_type='inner'):
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
    :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> lagrange_coefficients_ASTME2022(10, 'inner')
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
    >>> lagrange_coefficients_ASTME2022(10, 'boundary')
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

    global _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS
    if _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS is None:
        _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS = {}

    hash_key = tuple([hash(arg) for arg in (interval, interval_type)])
    if hash_key in _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS:
        return _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS[hash_key]

    r_n = np.linspace(1 / interval, 1 - (1 / interval), interval - 1)
    d = 3
    if interval_type.lower() == 'inner':
        r_n += 1
        d = 4

    lica = _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS[hash_key] = (
        as_float_array([lagrange_coefficients(r, d) for r in r_n]))

    return lica


def tristimulus_weighting_factors_ASTME2022(cmfs, illuminant, shape, k=None):
    """
    Returns a table of tristimulus weighting factors for given colour matching
    functions and illuminant using practise *ASTM E2022-11* method.

    The computed table of tristimulus weighting factors should be used with
    spectral data that has been corrected for spectral bandpass dependence.

    Parameters
    ----------
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    shape : SpectralShape
        Shape used to build the table, only the interval is needed.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.

    Returns
    -------
    ndarray
        Tristimulus weighting factors table.

    Raises
    ------
    ValueError
        If the colour matching functions or illuminant intervals are not equal
        to 1 nm.

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
    :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> from colour import (MSDS_CMFS, sd_CIE_standard_illuminant_A,
    ...     SpectralDistribution, SpectralShape)
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
    >>> A = sd_CIE_standard_illuminant_A(cmfs.shape)
    >>> with numpy_print_options(suppress=True):
    ...     tristimulus_weighting_factors_ASTME2022(
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

    global _CACHE_TRISTIMULUS_WEIGHTING_FACTORS
    if _CACHE_TRISTIMULUS_WEIGHTING_FACTORS is None:
        _CACHE_TRISTIMULUS_WEIGHTING_FACTORS = {}

    hash_key = tuple([
        hash(arg) for arg in (cmfs, illuminant, shape, k,
                              get_domain_range_scale())
    ])
    if hash_key in _CACHE_TRISTIMULUS_WEIGHTING_FACTORS:
        return _CACHE_TRISTIMULUS_WEIGHTING_FACTORS[hash_key]

    Y = cmfs.values
    S = illuminant.values

    interval_i = DEFAULT_INT_DTYPE(shape.interval)
    W = S[::interval_i, np.newaxis] * Y[::interval_i, :]

    # First and last measurement intervals *Lagrange Coefficients*.
    c_c = lagrange_coefficients_ASTME2022(interval_i, 'boundary')
    # Intermediate measurement intervals *Lagrange Coefficients*.
    c_b = lagrange_coefficients_ASTME2022(interval_i, 'inner')

    # Total wavelengths count.
    w_c = len(Y)
    # Measurement interval interpolated values count.
    r_c = c_b.shape[0]
    # Last interval first interpolated wavelength.
    w_lif = w_c - (w_c - 1) % interval_i - 1 - r_c

    # Intervals count.
    i_c = W.shape[0]
    i_cm = i_c - 1

    # "k" is used as index in the nested loop.
    k_n = k

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
        for j in range(
                DEFAULT_INT_DTYPE(w_c - ((w_c - 1) % interval_i)), w_c, 1):
            W[i_cm, i] = W[i_cm, i] + S[j] * Y[j, i]

    W *= 100 / np.sum(W, axis=0)[1] if k_n is None else k_n

    _CACHE_TRISTIMULUS_WEIGHTING_FACTORS[hash_key] = W

    return W


def adjust_tristimulus_weighting_factors_ASTME308(W, shape_r, shape_t):
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
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import (MSDS_CMFS, sd_CIE_standard_illuminant_A,
    ...     SpectralDistribution, SpectralShape)
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
    >>> A = sd_CIE_standard_illuminant_A(cmfs.shape)
    >>> W = tristimulus_weighting_factors_ASTME2022(
    ...     cmfs, A, SpectralShape(360, 830, 20))
    >>> with numpy_print_options(suppress=True):
    ...     adjust_tristimulus_weighting_factors_ASTME308(
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

    start_index = DEFAULT_INT_DTYPE(
        (shape_t.start - shape_r.start) / shape_r.interval)
    for i in range(start_index):
        W[start_index] += W[i]

    end_index = DEFAULT_INT_DTYPE(
        (shape_r.end - shape_t.end) / shape_r.interval)
    for i in range(end_index):
        W[-end_index - 1] += W[-i - 1]

    return W[start_index:-end_index or None, ...]


def sd_to_XYZ_integration(
        sd,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_DEFAULT),
        illuminant=sd_ones(),
        k=None):
    """
    Converts given spectral distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant according to classical
    integration method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution)
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
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
    >>> sd = SpectralDistribution(data)
    >>> illuminant = SDS_ILLUMINANTS['D65']
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 10.8404805...,   9.6838697...,   6.2115722...])
    """

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    if sd.shape != cmfs.shape:
        runtime_warning('Aligning "{0}" spectral distribution shape to "{1}" '
                        'colour matching functions shape.'.format(
                            sd.name, cmfs.name))
        sd = sd.copy().align(cmfs.shape)

    S = illuminant.values
    x_bar, y_bar, z_bar = tsplit(cmfs.values)
    R = sd.values

    dw = cmfs.shape.interval

    k = 100 / (np.sum(y_bar * S) * dw) if k is None else k

    X_p = R * x_bar * S * dw
    Y_p = R * y_bar * S * dw
    Z_p = R * z_bar * S * dw

    XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)

    return from_range_100(XYZ)


def sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
        sd,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_ASTME308),
        illuminant=sd_ones(SPECTRAL_SHAPE_ASTME308),
        k=None):
    """
    Converts given spectral distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant using a table of
    tristimulus weighting factors according to practise *ASTM E308-15* method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution)
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
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
    >>> sd = SpectralDistribution(data)
    >>> illuminant = SDS_ILLUMINANTS['D65']
    >>> sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
    ...     sd, cmfs, illuminant)  # doctest: +ELLIPSIS
    array([ 10.8405832...,   9.6844909...,   6.2155622...])
    """

    if cmfs.shape.interval != 1:
        runtime_warning('Interpolating "{0}" cmfs to 1nm interval.'.format(
            cmfs.name))
        cmfs = cmfs.copy().interpolate(SpectralShape(interval=1))

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    if sd.shape.boundaries != cmfs.shape.boundaries:
        runtime_warning('Trimming "{0}" spectral distribution shape to "{1}" '
                        'colour matching functions shape.'.format(
                            illuminant.name, cmfs.name))
        sd = sd.copy().trim(cmfs.shape)

    W = tristimulus_weighting_factors_ASTME2022(
        cmfs, illuminant,
        SpectralShape(cmfs.shape.start, cmfs.shape.end, sd.shape.interval), k)
    start_w = cmfs.shape.start
    end_w = cmfs.shape.start + sd.shape.interval * (W.shape[0] - 1)
    W = adjust_tristimulus_weighting_factors_ASTME308(
        W, SpectralShape(start_w, end_w, sd.shape.interval), sd.shape)
    R = sd.values

    XYZ = np.sum(W * R[..., np.newaxis], axis=0)

    return from_range_100(XYZ)


def sd_to_XYZ_ASTME308(
        sd,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_ASTME308),
        illuminant=sd_ones(SPECTRAL_SHAPE_ASTME308),
        use_practice_range=True,
        mi_5nm_omission_method=True,
        mi_20nm_interpolation_method=True,
        k=None):
    """
    Converts given spectral distribution to *CIE XYZ* tristimulus values using
    given colour matching functions and illuminant according to practise
    *ASTM E308-15* method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    use_practice_range : bool, optional
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        5 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        20 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution)
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
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
    >>> sd = SpectralDistribution(data)
    >>> illuminant = SDS_ILLUMINANTS['D65']
    >>> sd_to_XYZ_ASTME308(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 10.8401953...,   9.6841740...,   6.2158913...])
    """

    if sd.shape.interval not in (1, 5, 10, 20):
        raise ValueError(
            'Tristimulus values conversion from spectral data according to '
            'practise "ASTM E308-15" should be performed on spectral data '
            'with measurement interval of 1, 5, 10 or 20nm!')

    if use_practice_range:
        cmfs = cmfs.copy().trim(SPECTRAL_SHAPE_ASTME308)

    method = sd_to_XYZ_tristimulus_weighting_factors_ASTME308
    if sd.shape.interval == 1:
        method = sd_to_XYZ_integration
    elif sd.shape.interval == 5 and mi_5nm_omission_method:
        if cmfs.shape.interval != 5:
            cmfs = cmfs.copy().interpolate(SpectralShape(interval=5))
        method = sd_to_XYZ_integration
    elif sd.shape.interval == 20 and mi_20nm_interpolation_method:
        sd = sd.copy()
        if sd.shape.boundaries != cmfs.shape.boundaries:
            runtime_warning(
                'Trimming "{0}" spectral distribution shape to "{1}" '
                'colour matching functions shape.'.format(
                    illuminant.name, cmfs.name))
            sd.trim(cmfs.shape)

        # Extrapolation of additional 20nm padding intervals.
        sd.align(SpectralShape(sd.shape.start - 20, sd.shape.end + 20, 10))
        for i in range(2):
            sd[sd.wavelengths[i]] = (
                    3 * sd.values[i + 2] -
                    3 * sd.values[i + 4] + sd.values[i + 6])  # yapf: disable
            i_e = len(sd.domain) - 1 - i
            sd[sd.wavelengths[i_e]] = (
                sd.values[i_e - 6] - 3 * sd.values[i_e - 4] +
                3 * sd.values[i_e - 2])

        # Interpolating every odd numbered values.
        # TODO: Investigate code vectorisation.
        for i in range(3, len(sd.domain) - 3, 2):
            sd[sd.wavelengths[i]] = (
                -0.0625 * sd.values[i - 3] + 0.5625 * sd.values[i - 1] +
                0.5625 * sd.values[i + 1] - 0.0625 * sd.values[i + 3])

        # Discarding the additional 20nm padding intervals.
        sd.trim(SpectralShape(sd.shape.start + 20, sd.shape.end - 20, 10))

    XYZ = method(sd, cmfs, illuminant, k=k)

    return XYZ


SD_TO_XYZ_METHODS = CaseInsensitiveMapping({
    'ASTM E308': sd_to_XYZ_ASTME308,
    'Integration': sd_to_XYZ_integration
})
SD_TO_XYZ_METHODS.__doc__ = """
Supported spectral distribution to *CIE XYZ* tristimulus values conversion
methods.

References
----------
:cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
:cite:`Wyszecki2000bf`

SD_TO_XYZ_METHODS : CaseInsensitiveMapping
    **{'ASTM E308', 'Integration'}**

Aliases:

-   'astm2015': 'ASTM E308'
"""
SD_TO_XYZ_METHODS['astm2015'] = SD_TO_XYZ_METHODS['ASTM E308']


def sd_to_XYZ(
        sd,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_DEFAULT),
        illuminant=sd_ones(),
        k=None,
        method='ASTM E308',
        **kwargs):
    """
    Converts given spectral distribution to *CIE XYZ* tristimulus values using
    given colour matching functions, illuminant and method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.
    method : unicode, optional
        **{'ASTM E308', 'Integration'}**,
        Computation method.

    Other Parameters
    ----------------
    mi_5nm_omission_method : bool, optional
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        5 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        20 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    use_practice_range : bool, optional
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution)
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
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
    >>> sd = SpectralDistribution(data)
    >>> illuminant = SDS_ILLUMINANTS['D65']
    >>> sd_to_XYZ(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 10.8401953...,   9.6841740...,   6.2158913...])
    >>> sd_to_XYZ(sd, cmfs, illuminant, use_practice_range=False)
    ... # doctest: +ELLIPSIS
    array([ 10.8402774...,   9.6841967...,   6.2158838...])
    >>> sd_to_XYZ(sd, cmfs, illuminant, method='Integration')
    ... # doctest: +ELLIPSIS
    array([ 10.8404805...,   9.6838697...,   6.2115722...])
    """

    global _CACHE_SD_TO_XYZ
    if _CACHE_SD_TO_XYZ is None:
        _CACHE_SD_TO_XYZ = {}

    hash_key = tuple([
        hash(arg) for arg in (sd, cmfs, illuminant, k, method,
                              tuple(kwargs.items()), get_domain_range_scale())
    ])
    if hash_key in _CACHE_SD_TO_XYZ:
        return _CACHE_SD_TO_XYZ[hash_key]

    function = SD_TO_XYZ_METHODS[method]

    XYZ = _CACHE_SD_TO_XYZ[hash_key] = function(
        sd, cmfs, illuminant, k=k, **filter_kwargs(function, **kwargs))

    return XYZ


def msds_to_XYZ_integration(
        msds,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_DEFAULT),
        illuminant=sd_ones(),
        k=None,
        shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Converts given multi-spectral distributions to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant. The multi-spectral
    distribution can be either a :class:`colour.MultiSpectralDistributions`
    class instance or an *array_like* in which case the ``shape`` must be
    passed.

    Parameters
    ----------
    msds : MultiSpectralDistributions or array_like
        Multi-spectral distributions, if an *array_like* the wavelengths are
        expected to be in the last axis, e.g. for a 512x384 multi-spectral
        image with 77 bins, ``msds`` shape should be (384, 512, 77).
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.
    shape : SpectralShape, optional
        Spectral shape of the multi-spectral distributions, ``cmfs`` and
        ``illuminant`` will be aligned to it.

    Returns
    -------
    array_like
        *CIE XYZ* tristimulus values, for a 512x384 multi-spectral image with
        77 bins, the output shape will be (384, 512, 3).

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    -   The code path using the *array_like* multi-spectral distributions
        produces results different to the code path using a
        :class:`colour.MultiSpectralDistributions` class instance: the former
        favours execution speed by aligning the colour matching functions and
        illuminant to the given spectral shape while the latter favours
        precision by aligning the multi-spectral distributions to the colour
        matching functions.

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> shape = SpectralShape(400, 700, 60)
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> data = np.array([
    ...     [0.0137, 0.0159, 0.0096, 0.0111, 0.0179, 0.1057, 0.0433,
    ...      0.0258, 0.0248, 0.0186, 0.0310, 0.0473],
    ...     [0.0913, 0.3145, 0.2582, 0.0709, 0.2971, 0.4620, 0.2683,
    ...      0.0831, 0.1203, 0.1292, 0.1682, 0.3221],
    ...     [0.0152, 0.0842, 0.4139, 0.0220, 0.5630, 0.1918, 0.2373,
    ...      0.0430, 0.0054, 0.0079, 0.3719, 0.2268],
    ...     [0.0281, 0.0907, 0.2228, 0.1249, 0.2375, 0.5625, 0.0518,
    ...      0.3230, 0.0065, 0.4006, 0.0861, 0.3161],
    ...     [0.1918, 0.7103, 0.0041, 0.1817, 0.0024, 0.4209, 0.0118,
    ...      0.2302, 0.1860, 0.9404, 0.0041, 0.1124],
    ...     [0.0430, 0.0437, 0.3744, 0.0020, 0.5819, 0.0027, 0.0823,
    ...      0.0081, 0.3625, 0.3213, 0.7849, 0.0024],
    ... ])
    >>> msds = MultiSpectralDistributions(data, shape.range())
    >>> msds_to_XYZ_integration(msds, illuminant=D65, shape=shape)
    ... # doctest: +ELLIPSIS
    array([[  7.5029651...,   3.9487840...,   8.4034770...],
           [ 26.925986 ...,  15.0724738...,  28.7058153...],
           [ 16.7031140...,  28.2172235...,  25.6456293...],
           [ 11.5767146...,   8.6401095...,   6.5768486...],
           [ 18.7313077...,  35.0750086...,  30.1457629...],
           [ 45.1657291...,  39.6137391...,  43.6784025...],
           [  8.1755520...,  13.0934236...,  25.9421257...],
           [ 22.4676530...,  19.3099303...,   7.9637645...],
           [  6.5780111...,   2.5254943...,  11.0930902...],
           [ 43.9146821...,  27.9803874...,  11.7292796...],
           [  8.5363407...,  19.7029458...,  17.7051147...],
           [ 23.9088530...,  26.2129842...,  30.6763518...]])
    >>> msds = np.array([
    ...     [
    ...         [0.0137, 0.0913, 0.0152, 0.0281, 0.1918, 0.0430],
    ...         [0.0159, 0.3145, 0.0842, 0.0907, 0.7103, 0.0437],
    ...         [0.0096, 0.2582, 0.4139, 0.2228, 0.0041, 0.3744],
    ...         [0.0111, 0.0709, 0.0220, 0.1249, 0.1817, 0.0020],
    ...         [0.0179, 0.2971, 0.5630, 0.2375, 0.0024, 0.5819],
    ...         [0.1057, 0.4620, 0.1918, 0.5625, 0.4209, 0.0027],
    ...     ],
    ...     [
    ...         [0.0433, 0.2683, 0.2373, 0.0518, 0.0118, 0.0823],
    ...         [0.0258, 0.0831, 0.0430, 0.3230, 0.2302, 0.0081],
    ...         [0.0248, 0.1203, 0.0054, 0.0065, 0.1860, 0.3625],
    ...         [0.0186, 0.1292, 0.0079, 0.4006, 0.9404, 0.3213],
    ...         [0.0310, 0.1682, 0.3719, 0.0861, 0.0041, 0.7849],
    ...         [0.0473, 0.3221, 0.2268, 0.3161, 0.1124, 0.0024],
    ...     ],
    ... ])
    >>> msds_to_XYZ_integration(msds, illuminant=D65, shape=shape)
    ... # doctest: +ELLIPSIS
    array([[[  7.1958378...,   3.8605390...,  10.1016398...],
            [ 25.5738615...,  14.7200581...,  34.8440007...],
            [ 17.5854414...,  28.5668344...,  30.1806687...],
            [ 11.3271912...,   8.4598177...,   7.9015758...],
            [ 19.6581831...,  35.5918480...,  35.1430220...],
            [ 45.8212491...,  39.2600939...,  51.7907710...]],
    <BLANKLINE>
           [[  8.8287837...,  13.3870357...,  30.5702050...],
            [ 22.3324362...,  18.9560919...,   9.3952305...],
            [  6.6887212...,   2.5728891...,  13.2618778...],
            [ 41.8166227...,  27.1191979...,  14.2627944...],
            [  9.2414098...,  20.2056200...,  20.1992502...],
            [ 24.7830551...,  26.2221584...,  36.4430633...]]])
    """

    if isinstance(msds, MultiSpectralDistributions):
        return as_float_array([
            sd_to_XYZ_integration(sd, cmfs, illuminant, k)
            for sd in msds.to_sds()
        ])
    else:
        msds = as_float_array(msds)

        msd_shape_m_1, shape_wl_count = msds.shape[-1], len(shape.range())
        assert msd_shape_m_1 == shape_wl_count, (
            'Multi-spectral distributions array with {0} wavelengths '
            'is not compatible with spectral shape with {1} wavelengths!'.
            format(msd_shape_m_1, shape_wl_count))

        if cmfs.shape != shape:
            runtime_warning('Aligning "{0}" cmfs shape to "{1}".'.format(
                cmfs.name, shape))
            cmfs = cmfs.copy().align(shape)

        if illuminant.shape != shape:
            runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
                illuminant.name, shape))
            illuminant = illuminant.copy().align(shape)

        S = illuminant.values
        x_bar, y_bar, z_bar = tsplit(cmfs.values)
        dw = cmfs.shape.interval

        k = 100 / (np.sum(y_bar * S) * dw) if k is None else k

        X_p = msds * x_bar * S * dw
        Y_p = msds * y_bar * S * dw
        Z_p = msds * z_bar * S * dw

        XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)

        return from_range_100(np.rollaxis(XYZ, 0, msds.ndim))


def msds_to_XYZ_ASTME308(
        msds,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_ASTME308),
        illuminant=sd_ones(SPECTRAL_SHAPE_ASTME308),
        use_practice_range=True,
        mi_5nm_omission_method=True,
        mi_20nm_interpolation_method=True,
        k=None):
    """
    Converts given multi-spectral distributions to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant according to practise
    *ASTM E308-15* method.

    Parameters
    ----------
    msds : MultiSpectralDistributions or array_like
        Multi-spectral distributions.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    use_practice_range : bool, optional
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        5 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        20 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.
    shape : SpectralShape, optional
        Spectral shape of the multi-spectral distributions, ``cmfs`` and
        ``illuminant`` will be aligned to it.

    Returns
    -------
    array_like
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    -   The code path using the *array_like* multi-spectral distributions
        produces results different to the code path using a
        :class:`colour.MultiSpectralDistributions` class instance: the former
        favours execution speed by aligning the colour matching functions and
        illuminant to the given spectral shape while the latter favours
        precision by aligning the multi-spectral distributions to the colour
        matching functions.

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> shape = SpectralShape(400, 700, 60)
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> data = np.array([
    ...     [0.0137, 0.0159, 0.0096, 0.0111, 0.0179, 0.1057, 0.0433,
    ...      0.0258, 0.0248, 0.0186, 0.0310, 0.0473],
    ...     [0.0913, 0.3145, 0.2582, 0.0709, 0.2971, 0.4620, 0.2683,
    ...      0.0831, 0.1203, 0.1292, 0.1682, 0.3221],
    ...     [0.0152, 0.0842, 0.4139, 0.0220, 0.5630, 0.1918, 0.2373,
    ...      0.0430, 0.0054, 0.0079, 0.3719, 0.2268],
    ...     [0.0281, 0.0907, 0.2228, 0.1249, 0.2375, 0.5625, 0.0518,
    ...      0.3230, 0.0065, 0.4006, 0.0861, 0.3161],
    ...     [0.1918, 0.7103, 0.0041, 0.1817, 0.0024, 0.4209, 0.0118,
    ...      0.2302, 0.1860, 0.9404, 0.0041, 0.1124],
    ...     [0.0430, 0.0437, 0.3744, 0.0020, 0.5819, 0.0027, 0.0823,
    ...      0.0081, 0.3625, 0.3213, 0.7849, 0.0024],
    ... ])
    >>> msds = MultiSpectralDistributions(data, shape.range())
    >>> msds = msds.align(SpectralShape(400, 700, 20))
    >>> msds_to_XYZ_ASTME308(msds, illuminant=D65)
    ... # doctest: +ELLIPSIS
    array([[  7.5052758...,   3.9557516...,   8.38929  ...],
           [ 26.9408494...,  15.0987746...,  28.6631260...],
           [ 16.7047370...,  28.2089815...,  25.6556751...],
           [ 11.5711808...,   8.6445071...,   6.5587827...],
           [ 18.7428858...,  35.0626352...,  30.1778517...],
           [ 45.1224886...,  39.6238997...,  43.5813345...],
           [  8.1786985...,  13.0950215...,  25.9326459...],
           [ 22.4462888...,  19.3115133...,   7.9304333...],
           [  6.5764361...,   2.5305945...,  11.07253  ...],
           [ 43.9113380...,  28.0003541...,  11.6852531...],
           [  8.5496209...,  19.6913570...,  17.7400079...],
           [ 23.8866733...,  26.2147704...,  30.6297684...]])
    """

    if isinstance(msds, MultiSpectralDistributions):
        return as_float_array([
            sd_to_XYZ_ASTME308(sd, cmfs, illuminant, use_practice_range,
                               mi_5nm_omission_method,
                               mi_20nm_interpolation_method, k)
            for sd in msds.to_sds()
        ])
    else:
        raise ValueError('"ASTM E308-15" method does not support "array_like" '
                         'multi-spectral distributions!')


MSDS_TO_XYZ_METHODS = CaseInsensitiveMapping({
    'ASTM E308': msds_to_XYZ_ASTME308,
    'Integration': msds_to_XYZ_integration
})
MSDS_TO_XYZ_METHODS.__doc__ = """
Supported multi-spectral array to *CIE XYZ* tristimulus values conversion
methods.

References
----------
:cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
:cite:`Wyszecki2000bf`

MSDS_TO_XYZ_METHODS : CaseInsensitiveMapping
    **{'ASTM E308', 'Integration'}**

Aliases:

-   'astm2015': 'ASTM E308'
"""
MSDS_TO_XYZ_METHODS['astm2015'] = MSDS_TO_XYZ_METHODS['ASTM E308']


def msds_to_XYZ(
        msds,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().trim(SPECTRAL_SHAPE_DEFAULT),
        illuminant=sd_ones(),
        k=None,
        method='ASTM E308',
        **kwargs):
    """
    Converts given multi-spectral distributions to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant. For the *Integration*
    method, the multi-spectral distributions can be either a
    :class:`colour.MultiSpectralDistributions` class instance or an
    *array_like* in which case the ``shape`` must be passed.

    Parameters
    ----------
    msds : MultiSpectralDistributions or array_like
        Multi-spectral distributions, if an *array_like* the wavelengths are
        expected to be in the last axis, e.g. for a 512x384 multi-spectral
        image with 77 bins, ``msds`` shape should be (384, 512, 77).
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    k : numeric, optional
        Normalisation constant :math:`k`. For reflecting or transmitting object
        colours, :math:`k` is chosen so that :math:`Y = 100` for objects for
        which the spectral reflectance factor :math:`R(\\lambda)` of the object
        colour or the spectral transmittance factor :math:`\\tau(\\lambda)` of
        the object is equal to unity for all wavelengths. For self-luminous
        objects and illuminants, the constants :math:`k` is usually chosen on
        the grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be numerically
        equal to the absolute value of a photometric quantity, the constant,
        :math:`k`, must be put equal to the numerical value of :math:`K_m`, the
        maximum spectral luminous efficacy (which is equal to
        683 :math:`lm\\cdot W^{-1}`) and :math:`\\Phi_\\lambda(\\lambda)` must
        be the spectral concentration of the radiometric quantity corresponding
        to the photometric quantity required.
    method : unicode, optional
        **{'ASTM E308', 'Integration'}**,
        Computation method.

    Other Parameters
    ----------------
    use_practice_range : bool, optional
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method : bool, optional
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        5 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method : bool, optional
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        20 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    shape : SpectralShape, optional
        {:func:`colour.colorimetry.msds_to_XYZ_integration`},
        Spectral shape of the multi-spectral distributions array :math:`msds`,
        ``cmfs`` and ``illuminant`` will be aligned to it.

    Returns
    -------
    array_like
        *CIE XYZ* tristimulus values, for a 512x384 multi-spectral image with
        77 wavelengths, the output shape will be (384, 512, 3).

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 100]              | [0, 1]        |
    +-----------+-----------------------+---------------+

    -   The code path using the *array_like* multi-spectral distributions
        produces results different to the code path using a
        :class:`colour.MultiSpectralDistributions` class instance: the former
        favours execution speed by aligning the colour matching functions and
        illuminant to the given spectral shape while the latter favours
        precision by aligning the multi-spectral distributions to the colour
        matching functions.

    References
    ----------
    :cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> shape = SpectralShape(400, 700, 60)
    >>> data = np.array([
    ...     [0.0137, 0.0159, 0.0096, 0.0111, 0.0179, 0.1057, 0.0433,
    ...      0.0258, 0.0248, 0.0186, 0.0310, 0.0473],
    ...     [0.0913, 0.3145, 0.2582, 0.0709, 0.2971, 0.4620, 0.2683,
    ...      0.0831, 0.1203, 0.1292, 0.1682, 0.3221],
    ...     [0.0152, 0.0842, 0.4139, 0.0220, 0.5630, 0.1918, 0.2373,
    ...      0.0430, 0.0054, 0.0079, 0.3719, 0.2268],
    ...     [0.0281, 0.0907, 0.2228, 0.1249, 0.2375, 0.5625, 0.0518,
    ...      0.3230, 0.0065, 0.4006, 0.0861, 0.3161],
    ...     [0.1918, 0.7103, 0.0041, 0.1817, 0.0024, 0.4209, 0.0118,
    ...      0.2302, 0.1860, 0.9404, 0.0041, 0.1124],
    ...     [0.0430, 0.0437, 0.3744, 0.0020, 0.5819, 0.0027, 0.0823,
    ...      0.0081, 0.3625, 0.3213, 0.7849, 0.0024],
    ... ])
    >>> msds = MultiSpectralDistributions(data, shape.range())
    >>> msds_to_XYZ(msds, method='Integration', shape=shape)
    ... # doctest: +ELLIPSIS
    array([[  8.2415862...,   4.2543993...,   7.6100842...],
           [ 29.6144619...,  16.1158465...,  25.9015472...],
           [ 16.6799560...,  27.2350547...,  22.9413337...],
           [ 12.5597688...,   9.0667136...,   5.9670327...],
           [ 18.5804689...,  33.6618109...,  26.9249733...],
           [ 47.7113308...,  40.4573249...,  39.6439145...],
           [  7.830207 ...,  12.3689624...,  23.3742655...],
           [ 24.1695370...,  20.0629815...,   7.2718670...],
           [  7.2333751...,   2.7982097...,  10.0688374...],
           [ 48.7358074...,  30.2417164...,  10.6753233...],
           [  8.3231013...,  18.6791507...,  15.8228184...],
           [ 24.6452277...,  26.0809382...,  27.7106399...]])
    >>> msds = np.array([
    ...     [
    ...         [0.0137, 0.0913, 0.0152, 0.0281, 0.1918, 0.0430],
    ...         [0.0159, 0.3145, 0.0842, 0.0907, 0.7103, 0.0437],
    ...         [0.0096, 0.2582, 0.4139, 0.2228, 0.0041, 0.3744],
    ...         [0.0111, 0.0709, 0.0220, 0.1249, 0.1817, 0.0020],
    ...         [0.0179, 0.2971, 0.5630, 0.2375, 0.0024, 0.5819],
    ...         [0.1057, 0.4620, 0.1918, 0.5625, 0.4209, 0.0027],
    ...     ],
    ...     [
    ...         [0.0433, 0.2683, 0.2373, 0.0518, 0.0118, 0.0823],
    ...         [0.0258, 0.0831, 0.0430, 0.3230, 0.2302, 0.0081],
    ...         [0.0248, 0.1203, 0.0054, 0.0065, 0.1860, 0.3625],
    ...         [0.0186, 0.1292, 0.0079, 0.4006, 0.9404, 0.3213],
    ...         [0.0310, 0.1682, 0.3719, 0.0861, 0.0041, 0.7849],
    ...         [0.0473, 0.3221, 0.2268, 0.3161, 0.1124, 0.0024],
    ...     ],
    ... ])
    >>> msds_to_XYZ(msds, method='Integration', shape=shape)
    ... # doctest: +ELLIPSIS
    array([[[  7.6862675...,   4.0925470...,   8.4950412...],
            [ 27.4119366...,  15.5014764...,  29.2825122...],
            [ 17.1283666...,  27.7798651...,  25.5232032...],
            [ 11.9824544...,   8.8127109...,   6.6518695...],
            [ 19.1030682...,  34.4597818...,  29.7653804...],
            [ 46.8243374...,  39.9551652...,  43.6541858...]],
    <BLANKLINE>
           [[  8.0978189...,  12.7544378...,  25.8004512...],
            [ 23.4360673...,  19.6127966...,   7.9342408...],
            [  7.0933208...,   2.7894394...,  11.1527704...],
            [ 45.6313772...,  29.0068105...,  11.9934522...],
            [  8.9327884...,  19.4008147...,  17.1534186...],
            [ 24.6610235...,  26.1093760...,  30.7298791...]]])
    """

    function = MSDS_TO_XYZ_METHODS[method]

    return function(msds, cmfs, illuminant, k,
                    **filter_kwargs(function, **kwargs))


def wavelength_to_XYZ(wavelength,
                      cmfs=MSDS_CMFS_STANDARD_OBSERVER[
                          'CIE 1931 2 Degree Standard Observer']):
    """
    Converts given wavelength :math:`\\lambda` to *CIE XYZ* tristimulus values
    using given colour matching functions.

    If the wavelength :math:`\\lambda` is not available in the colour matching
    function, its value will be calculated according to *CIE 15:2004*
    recommendation: the method developed by *Sprague (1880)* will be used for
    interpolating functions having a uniformly spaced independent variable and
    the *Cubic Spline* method for non-uniformly spaced independent variable.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in nm.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If wavelength :math:`\\lambda` is not contained in the colour matching
        functions domain.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [0, 1]                | [0, 1]        |
    +-----------+-----------------------+---------------+

    Examples
    --------
    >>> from colour import MSDS_CMFS
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> wavelength_to_XYZ(480, cmfs)  # doctest: +ELLIPSIS
    array([ 0.09564  ,  0.13902  ,  0.8129501...])
    >>> wavelength_to_XYZ(480.5, cmfs)  # doctest: +ELLIPSIS
    array([ 0.0914287...,  0.1418350...,  0.7915726...])
    """

    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
            np.max(wavelength) > cmfs_shape.end):
        raise ValueError(
            '"{0}nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, cmfs_shape.start, cmfs_shape.end))

    XYZ = np.reshape(cmfs[np.ravel(wavelength)],
                     as_float_array(wavelength).shape + (3, ))

    return XYZ
