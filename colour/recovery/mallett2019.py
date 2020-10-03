# -*- coding: utf-8 -*-
"""
Mallett and Yuksel (2019) - Reflectance Recovery
================================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Mallett and Yuksel (2019)* method:

-   :func:`colour.recovery.spectral_primary_decomposition_Mallett2019`
-   :func:`colour.recovery.RGB_to_sd_Mallett2019`

References
----------
-   :cite:`Mallett2019` : Mallett, I., & Yuksel, C. (2019). Spectral Primary
    Decomposition for Rendering with sRGB Reflectance. Eurographics Symposium
    on Rendering - DL-Only and Industry Track, 7 pages. doi:10.2312/SR.20191216
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from colour.colorimetry import (SpectralDistribution,
                                MultiSpectralDistributions,
                                MSDS_CMFS_STANDARD_OBSERVER, SDS_ILLUMINANTS)
from colour.recovery import MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
from colour.utilities import to_domain_1, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'spectral_primary_decomposition_Mallett2019',
    'RGB_to_sd_Mallett2019',
]


def spectral_primary_decomposition_Mallett2019(
        colourspace,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER[
            'CIE 1931 2 Degree Standard Observer'],
        illuminant=SDS_ILLUMINANTS['D65'],
        metric=np.linalg.norm,
        metric_args=tuple(),
        optimisation_kwargs=None):
    """
    Performs the spectral primary decomposition as described in *Mallett and
    Yuksel (2019)* for given *RGB* colourspace.

    Parameters
    ----------
    colourspace: RGB_Colourspace
        *RGB* colourspace.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    metric : unicode, optional
        Function to be minimised, i.e. the objective function.

            ``metric(basis, *metric_args) -> float``

        where ``basis`` is three reflectances concatenated together, each
        with a shape matching ``shape``.
    metric_args : tuple, optional
        Additional arguments passed to ``metric``.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    MultiSpectralDistributions
        Basis functions for given *RGB* colourspace.

    References
    ----------
    :cite:`Mallett2019`

    Notes
    -----
    -   In-addition to the *BT.709* primaries used by the *sRGB* colourspace,
        :cite:`Mallett2019` tried *BT.2020*, *P3 D65*, *Adobe RGB 1998*,
        *NTSC (1987)*, *Pal/Secam*, *ProPhoto RGB*,
        and *Adobe Wide Gamut RGB* primaries, every one of which encompasses a
        larger (albeit not-always-enveloping) set of *CIE L\\*a\\*b\\** colours
        than BT.709. Of these, only *Pal/Secam* produces a feasible basis,
        which is relatively unsurprising since it is very similar to *BT.709*,
        whereas the others are significantly larger.

    Examples
    --------
    >>> from colour.colorimetry import SpectralShape
    >>> from colour.models import RGB_COLOURSPACE_PAL_SECAM
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> msds = spectral_primary_decomposition_Mallett2019(
    ...     RGB_COLOURSPACE_PAL_SECAM, cmfs, illuminant, optimisation_kwargs={
    ...         'options': {'ftol': 1e-5}
    ...     }
    ... )
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     print(msds)  # doctest: +SKIP
    [[ 360.            0.3324728...    0.3332663...    0.3342608...]
     [ 370.            0.3323307...    0.3327746...    0.3348946...]
     [ 380.            0.3341115...    0.3323995...    0.3334889...]
     [ 390.            0.3337570...    0.3298092...    0.3364336...]
     [ 400.            0.3209352...    0.3218213...    0.3572433...]
     [ 410.            0.2881025...    0.2837628...    0.4281346...]
     [ 420.            0.1836749...    0.1838893...    0.6324357...]
     [ 430.            0.0187212...    0.0529655...    0.9283132...]
     [ 440.            0.       ...    0.       ...    1.       ...]
     [ 450.            0.       ...    0.       ...    1.       ...]
     [ 460.            0.       ...    0.       ...    1.       ...]
     [ 470.            0.       ...    0.0509556...    0.9490443...]
     [ 480.            0.       ...    0.2933996...    0.7066003...]
     [ 490.            0.       ...    0.5001396...    0.4998603...]
     [ 500.            0.       ...    0.6734805...    0.3265195...]
     [ 510.            0.       ...    0.8555213...    0.1444786...]
     [ 520.            0.       ...    0.9999985...    0.0000014...]
     [ 530.            0.       ...    1.       ...    0.       ...]
     [ 540.            0.       ...    1.       ...    0.       ...]
     [ 550.            0.       ...    1.       ...    0.       ...]
     [ 560.            0.       ...    0.9924229...    0.       ...]
     [ 570.            0.       ...    0.9913344...    0.0083032...]
     [ 580.            0.0370289...    0.9145168...    0.0484542...]
     [ 590.            0.7100075...    0.2898477...    0.0001446...]
     [ 600.            1.       ...    0.       ...    0.       ...]
     [ 610.            1.       ...    0.       ...    0.       ...]
     [ 620.            1.       ...    0.       ...    0.       ...]
     [ 630.            1.       ...    0.       ...    0.       ...]
     [ 640.            0.9711347...    0.0137659...    0.0150993...]
     [ 650.            0.7996619...    0.1119379...    0.0884001...]
     [ 660.            0.6064640...    0.202815 ...    0.1907209...]
     [ 670.            0.4662959...    0.2675037...    0.2662005...]
     [ 680.            0.4010958...    0.2998989...    0.2990052...]
     [ 690.            0.3617485...    0.3208921...    0.3173592...]
     [ 700.            0.3496691...    0.3247855...    0.3255453...]
     [ 710.            0.3433979...    0.3273540...    0.329248 ...]
     [ 720.            0.3358860...    0.3345583...    0.3295556...]
     [ 730.            0.3349498...    0.3314232...    0.3336269...]
     [ 740.            0.3359954...    0.3340147...    0.3299897...]
     [ 750.            0.3310392...    0.3327595...    0.3362012...]
     [ 760.            0.3346883...    0.3314158...    0.3338957...]
     [ 770.            0.3332167...    0.333371 ...    0.3334122...]
     [ 780.            0.3319670...    0.3325476...    0.3354852...]]
    """

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    N = len(cmfs.shape)

    R_to_XYZ = np.transpose(
        np.expand_dims(illuminant.values, axis=1) * cmfs.values / (np.sum(
            cmfs.values[:, 1] * illuminant.values)))
    R_to_RGB = np.dot(colourspace.matrix_XYZ_to_RGB, R_to_XYZ)
    basis_to_RGB = block_diag(R_to_RGB, R_to_RGB, R_to_RGB)

    primaries = np.identity(3).reshape(9)

    # Ensure the reflectances correspond to the correct RGB colours.
    colour_match = LinearConstraint(basis_to_RGB, primaries, primaries)

    # Ensure the reflectances are bounded by [0, 1].
    energy_conservation = Bounds(np.zeros(3 * N), np.ones(3 * N))

    # Ensure the sum of the three bases is bounded by [0, 1].
    sum_matrix = np.transpose(np.tile(np.identity(N), (3, 1)))
    sum_constraint = LinearConstraint(sum_matrix, np.zeros(N), np.ones(N))

    optimisation_settings = {
        'method': 'SLSQP',
        'constraints': [colour_match, sum_constraint],
        'bounds': energy_conservation,
        'options': {
            'ftol': 1e-10,
        }
    }

    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(
        metric, args=metric_args, x0=np.zeros(3 * N), **optimisation_settings)

    basis_functions = np.transpose(result.x.reshape(3, N))

    return MultiSpectralDistributions(
        basis_functions,
        cmfs.shape.range(),
        name='Basis Functions - {0} - Mallett (2019)'.format(colourspace.name),
        labels=('red', 'green', 'blue'))


def RGB_to_sd_Mallett2019(
        RGB, basis_functions=MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019):
    """
    Recovers the spectral distribution of given *RGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array.
    basis_functions : MultiSpectralDistributions
        Basis functions for the method. The default is to use the built-in
        *sRGB* basis functions, i.e.
        :attr:`colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019`.

    Returns
    -------
    SpectralDistribution
        Recovered reflectance.

    References
    ----------
    :cite:`Mallett2019`

    Notes
    -----
    -   In-addition to the *BT.709* primaries used by the *sRGB* colourspace,
        :cite:`Mallett2019` tried *BT.2020*, *P3 D65*, *Adobe RGB 1998*,
        *NTSC (1987)*, *Pal/Secam*, *ProPhoto RGB*,
        and *Adobe Wide Gamut RGB* primaries, every one of which encompasses a
        larger (albeit not-always-enveloping) set of *CIE L\\*a\\*b\\** colours
        than BT.709. Of these, only *Pal/Secam* produces a feasible basis,
        which is relatively unsurprising since it is very similar to *BT.709*,
        whereas the others are significantly larger.

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS, sd_to_XYZ_integration
    >>> from colour.models import XYZ_to_sRGB
    >>> from colour.recovery import SPECTRAL_SHAPE_sRGB_MALLETT2019
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SPECTRAL_SHAPE_sRGB_MALLETT2019)
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Mallett2019(RGB)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 380.        ,    0.1735531...],
                          [ 385.        ,    0.1720357...],
                          [ 390.        ,    0.1677721...],
                          [ 395.        ,    0.1576605...],
                          [ 400.        ,    0.1372829...],
                          [ 405.        ,    0.1170849...],
                          [ 410.        ,    0.0895694...],
                          [ 415.        ,    0.0706232...],
                          [ 420.        ,    0.0585765...],
                          [ 425.        ,    0.0523959...],
                          [ 430.        ,    0.0497598...],
                          [ 435.        ,    0.0476057...],
                          [ 440.        ,    0.0465079...],
                          [ 445.        ,    0.0460337...],
                          [ 450.        ,    0.0455839...],
                          [ 455.        ,    0.0452872...],
                          [ 460.        ,    0.0450981...],
                          [ 465.        ,    0.0448895...],
                          [ 470.        ,    0.0449257...],
                          [ 475.        ,    0.0448987...],
                          [ 480.        ,    0.0446834...],
                          [ 485.        ,    0.0441372...],
                          [ 490.        ,    0.0417137...],
                          [ 495.        ,    0.0373832...],
                          [ 500.        ,    0.0357657...],
                          [ 505.        ,    0.0348263...],
                          [ 510.        ,    0.0341953...],
                          [ 515.        ,    0.0337683...],
                          [ 520.        ,    0.0334979...],
                          [ 525.        ,    0.0332991...],
                          [ 530.        ,    0.0331909...],
                          [ 535.        ,    0.0332181...],
                          [ 540.        ,    0.0333387...],
                          [ 545.        ,    0.0334970...],
                          [ 550.        ,    0.0337381...],
                          [ 555.        ,    0.0341847...],
                          [ 560.        ,    0.0346447...],
                          [ 565.        ,    0.0353993...],
                          [ 570.        ,    0.0367367...],
                          [ 575.        ,    0.0392007...],
                          [ 580.        ,    0.0445902...],
                          [ 585.        ,    0.0625633...],
                          [ 590.        ,    0.2965381...],
                          [ 595.        ,    0.4215576...],
                          [ 600.        ,    0.4347139...],
                          [ 605.        ,    0.4385134...],
                          [ 610.        ,    0.4385184...],
                          [ 615.        ,    0.4385249...],
                          [ 620.        ,    0.4374694...],
                          [ 625.        ,    0.4384672...],
                          [ 630.        ,    0.4368251...],
                          [ 635.        ,    0.4340867...],
                          [ 640.        ,    0.4303219...],
                          [ 645.        ,    0.4243257...],
                          [ 650.        ,    0.4159482...],
                          [ 655.        ,    0.4057443...],
                          [ 660.        ,    0.3919874...],
                          [ 665.        ,    0.3742784...],
                          [ 670.        ,    0.3518421...],
                          [ 675.        ,    0.3240127...],
                          [ 680.        ,    0.2955145...],
                          [ 685.        ,    0.2625658...],
                          [ 690.        ,    0.2343423...],
                          [ 695.        ,    0.2174830...],
                          [ 700.        ,    0.2060461...],
                          [ 705.        ,    0.1977437...],
                          [ 710.        ,    0.1916846...],
                          [ 715.        ,    0.1861020...],
                          [ 720.        ,    0.1823908...],
                          [ 725.        ,    0.1807923...],
                          [ 730.        ,    0.1795571...],
                          [ 735.        ,    0.1785623...],
                          [ 740.        ,    0.1775758...],
                          [ 745.        ,    0.1771614...],
                          [ 750.        ,    0.1767431...],
                          [ 755.        ,    0.1764319...],
                          [ 760.        ,    0.1762597...],
                          [ 765.        ,    0.1762209...],
                          [ 770.        ,    0.1761803...],
                          [ 775.        ,    0.1761195...],
                          [ 780.        ,    0.1760763...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100
    ... # doctest: +ELLIPSIS
    array([ 0.2065436...,  0.1219996...,  0.0513764...])
    """

    RGB = to_domain_1(RGB)

    sd = SpectralDistribution(
        np.dot(RGB, np.transpose(basis_functions.values)),
        basis_functions.wavelengths)
    sd.name = '{0} (RGB) - Mallett (2019)'.format(RGB)

    return sd
