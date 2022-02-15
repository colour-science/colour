"""
Meng et al. (2015) - Reflectance Recovery
=========================================

Defines the objects for reflectance recovery using
*Meng, Simon and Hanika (2015)* method:

-   :func:`colour.recovery.XYZ_to_sd_Meng2015`

References
----------
-   :cite:`Meng2015c` : Meng, J., Simon, F., Hanika, J., & Dachsbacher, C.
    (2015). Physically Meaningful Rendering using Tristimulus Colours. Computer
    Graphics Forum, 34(4), 31-40. doi:10.1111/cgf.12676
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    sd_ones,
    sd_to_XYZ_integration,
)
from colour.hints import ArrayLike, Dict, FloatingOrNDArray, NDArray, Optional
from colour.utilities import to_domain_1, from_range_100

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_MENG2015",
    "XYZ_to_sd_Meng2015",
]

SPECTRAL_SHAPE_MENG2015: SpectralShape = SpectralShape(360, 780, 5)
"""
Spectral shape according to *ASTM E308-15* practise shape but using an interval
of 5.
"""


def XYZ_to_sd_Meng2015(
    XYZ: ArrayLike,
    cmfs: Optional[MultiSpectralDistributions] = None,
    illuminant: Optional[SpectralDistribution] = None,
    optimisation_kwargs: Optional[Dict] = None,
) -> SpectralDistribution:
    """
    Recover the spectral distribution of given *CIE XYZ* tristimulus values
    using *Meng et al. (2015)* method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs
        Standard observer colour matching functions. The wavelength
        :math:`\\lambda_{i}` range interval of the colour matching functions
        affects directly the time the computations take. The current default
        interval of 5 is a good compromise between precision and time spent,
        default to the *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Recovered spectral distribution.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The definition used to convert spectrum to *CIE XYZ* tristimulus
        values is :func:`colour.colorimetry.spectral_to_XYZ_integration`
        definition because it processes any measurement interval opposed to
        :func:`colour.colorimetry.sd_to_XYZ_ASTME308` definition that
        handles only measurement interval of 1, 5, 10 or 20nm.

    References
    ----------
    :cite:`Meng2015c`

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd_Meng2015(XYZ, cmfs, illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.0762005...],
                          [ 370.        ,    0.0761792...],
                          [ 380.        ,    0.0761363...],
                          [ 390.        ,    0.0761194...],
                          [ 400.        ,    0.0762539...],
                          [ 410.        ,    0.0761671...],
                          [ 420.        ,    0.0754649...],
                          [ 430.        ,    0.0731519...],
                          [ 440.        ,    0.0676701...],
                          [ 450.        ,    0.0577800...],
                          [ 460.        ,    0.0441993...],
                          [ 470.        ,    0.0285064...],
                          [ 480.        ,    0.0138728...],
                          [ 490.        ,    0.0033585...],
                          [ 500.        ,    0.       ...],
                          [ 510.        ,    0.       ...],
                          [ 520.        ,    0.       ...],
                          [ 530.        ,    0.       ...],
                          [ 540.        ,    0.0055767...],
                          [ 550.        ,    0.0317581...],
                          [ 560.        ,    0.0754491...],
                          [ 570.        ,    0.1314115...],
                          [ 580.        ,    0.1937649...],
                          [ 590.        ,    0.2559311...],
                          [ 600.        ,    0.3123173...],
                          [ 610.        ,    0.3584966...],
                          [ 620.        ,    0.3927335...],
                          [ 630.        ,    0.4159458...],
                          [ 640.        ,    0.4306660...],
                          [ 650.        ,    0.4391040...],
                          [ 660.        ,    0.4439497...],
                          [ 670.        ,    0.4463618...],
                          [ 680.        ,    0.4474625...],
                          [ 690.        ,    0.4479868...],
                          [ 700.        ,    0.4482116...],
                          [ 710.        ,    0.4482800...],
                          [ 720.        ,    0.4483472...],
                          [ 730.        ,    0.4484251...],
                          [ 740.        ,    0.4484633...],
                          [ 750.        ,    0.4485071...],
                          [ 760.        ,    0.4484969...],
                          [ 770.        ,    0.4484853...],
                          [ 780.        ,    0.4485134...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    XYZ = to_domain_1(XYZ)

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, shape_default=SPECTRAL_SHAPE_MENG2015
    )

    sd = sd_ones(cmfs.shape)

    def objective_function(a: ArrayLike) -> FloatingOrNDArray:
        """Define the objective function."""

        return np.sum(np.diff(a) ** 2)

    def constraint_function(a: ArrayLike) -> NDArray:
        """Define the constraint function."""

        sd[:] = a
        return (
            sd_to_XYZ_integration(sd, cmfs=cmfs, illuminant=illuminant) - XYZ
        )

    wavelengths = sd.wavelengths
    bins = wavelengths.size

    optimisation_settings = {
        "method": "SLSQP",
        "constraints": {"type": "eq", "fun": constraint_function},
        "bounds": np.tile(np.array([0, 1000]), (bins, 1)),
        "options": {
            "ftol": 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(objective_function, sd.values, **optimisation_settings)

    if not result.success:
        raise RuntimeError(
            f"Optimization failed for {XYZ} after {result.nit} iterations: "
            f'"{result.message}".'
        )

    return SpectralDistribution(
        from_range_100(result.x * 100),
        wavelengths,
        name=f"{XYZ} (XYZ) - Meng (2015)",
    )
