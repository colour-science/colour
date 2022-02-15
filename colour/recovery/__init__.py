"""
References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147-155. doi:10.1111/cgf.13626
-   :cite:`Mallett2019` : Mallett, I., & Yuksel, C. (2019). Spectral Primary
    Decomposition for Rendering with sRGB Reflectance. Eurographics Symposium
    on Rendering - DL-Only and Industry Track, 7 pages. doi:10.2312/SR.20191216
-   :cite:`Meng2015c` : Meng, J., Simon, F., Hanika, J., & Dachsbacher, C.
    (2015). Physically Meaningful Rendering using Tristimulus Colours. Computer
    Graphics Forum, 34(4), 31-40. doi:10.1111/cgf.12676
-   :cite:`Otsu2018` : Otsu, H., Yamamoto, M., & Hachisuka, T. (2018).
    Reproducing Spectral Reflectances From Tristimulus Colours. Computer
    Graphics Forum, 37(6), 370-381. doi:10.1111/cgf.13332
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import annotations

import sys

from colour.colorimetry import SpectralDistribution
from colour.hints import Any, ArrayLike, Literal, Union
from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    filter_kwargs,
    validate_method,
)

from .datasets import *  # noqa
from . import datasets
from .jakob2019 import (
    sd_Jakob2019,
    find_coefficients_Jakob2019,
    XYZ_to_sd_Jakob2019,
    LUT3D_Jakob2019,
)
from .mallett2019 import (
    spectral_primary_decomposition_Mallett2019,
    RGB_to_sd_Mallett2019,
)
from .meng2015 import XYZ_to_sd_Meng2015
from .otsu2018 import Dataset_Otsu2018, Tree_Otsu2018, XYZ_to_sd_Otsu2018
from .smits1999 import RGB_to_sd_Smits1999

__all__ = []
__all__ += datasets.__all__
__all__ += [
    "sd_Jakob2019",
    "find_coefficients_Jakob2019",
    "XYZ_to_sd_Jakob2019",
    "LUT3D_Jakob2019",
]
__all__ += [
    "spectral_primary_decomposition_Mallett2019",
    "RGB_to_sd_Mallett2019",
]
__all__ += [
    "XYZ_to_sd_Meng2015",
]
__all__ += [
    "Dataset_Otsu2018",
    "Tree_Otsu2018",
    "XYZ_to_sd_Otsu2018",
]
__all__ += [
    "RGB_to_sd_Smits1999",
]

XYZ_TO_SD_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Jakob 2019": XYZ_to_sd_Jakob2019,
        "Mallett 2019": RGB_to_sd_Mallett2019,
        "Meng 2015": XYZ_to_sd_Meng2015,
        "Otsu 2018": XYZ_to_sd_Otsu2018,
        "Smits 1999": RGB_to_sd_Smits1999,
    }
)
XYZ_TO_SD_METHODS.__doc__ = """
Supported spectral distribution recovery methods.

References
----------
:cite:`Jakob2019`, :cite:`Mallett2019`, :cite:`Meng2015c`,
:cite:`Smits1999a`
"""


def XYZ_to_sd(
    XYZ: ArrayLike,
    method: Union[
        Literal[
            "Jakob 2019",
            "Mallett 2019",
            "Meng 2015",
            "Otsu 2018",
            "Smits 1999",
        ],
        str,
    ] = "Meng 2015",
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Recover the spectral distribution of given *CIE XYZ* tristimulus
    values using given method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values to recover the spectral distribution
        from.
    method
        Computation method.

    Other Parameters
    ----------------
    additional_data
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`},
        If *True*, ``error`` will be returned alongside ``sd``.
    basis_functions
        {:func:`colour.recovery.RGB_to_sd_Mallett2019`},
        Basis functions for the method. The default is to use the built-in
        *sRGB* basis functions, i.e.
        :attr:`colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019`.
    clip
        {:func:`colour.recovery.XYZ_to_sd_Otsu2018`},
        If *True*, the default, values below zero and above unity in the
        recovered spectral distributions will be clipped. This ensures that the
        returned reflectance is physical and conserves energy, but will cause
        noticeable colour differences in case of very saturated colours.
    cmfs
        {:func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Standard observer colour matching functions.
    colourspace
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`},
        *RGB* colourspace of the target colour. Note that no chromatic
        adaptation is performed between ``illuminant`` and the colourspace
        whitepoint.
    dataset
        {:func:`colour.recovery.XYZ_to_sd_Otsu2018`},
        Dataset to use for reconstruction. The default is to use the published
        data.
    illuminant
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`,
        :func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    interval
        {:func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Wavelength :math:`\\lambda_{i}` range interval in nm. The smaller
        ``interval`` is, the longer the computations will be.
    optimisation_kwargs
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`,
        :func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Parameters for :func:`scipy.optimize.minimize` and
        :func:`colour.recovery.find_coefficients_Jakob2019` definitions.

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

    -   *Smits (1999)* method will internally convert given *CIE XYZ*
        tristimulus values to *sRGB* colourspace array assuming equal energy
        illuminant *E*.

    References
    ----------
    :cite:`Jakob2019`,  :cite:`Mallett2019`, :cite:`Meng2015c`,
    :cite:`Otsu2018`, :cite:`Smits1999a`

    Examples
    --------
    *Jakob and Hanika (2009)* reflectance recovery:

    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd(
    ...     XYZ, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 360.        ,    0.4893773...],
                          [ 370.        ,    0.3258214...],
                          [ 380.        ,    0.2147792...],
                          [ 390.        ,    0.1482413...],
                          [ 400.        ,    0.1086169...],
                          [ 410.        ,    0.0841255...],
                          [ 420.        ,    0.0683114...],
                          [ 430.        ,    0.0577144...],
                          [ 440.        ,    0.0504267...],
                          [ 450.        ,    0.0453552...],
                          [ 460.        ,    0.0418520...],
                          [ 470.        ,    0.0395259...],
                          [ 480.        ,    0.0381430...],
                          [ 490.        ,    0.0375741...],
                          [ 500.        ,    0.0377685...],
                          [ 510.        ,    0.0387432...],
                          [ 520.        ,    0.0405871...],
                          [ 530.        ,    0.0434783...],
                          [ 540.        ,    0.0477225...],
                          [ 550.        ,    0.0538256...],
                          [ 560.        ,    0.0626314...],
                          [ 570.        ,    0.0755869...],
                          [ 580.        ,    0.0952675...],
                          [ 590.        ,    0.1264265...],
                          [ 600.        ,    0.1779272...],
                          [ 610.        ,    0.2649393...],
                          [ 620.        ,    0.4039779...],
                          [ 630.        ,    0.5832105...],
                          [ 640.        ,    0.7445440...],
                          [ 650.        ,    0.8499970...],
                          [ 660.        ,    0.9094792...],
                          [ 670.        ,    0.9425378...],
                          [ 680.        ,    0.9616376...],
                          [ 690.        ,    0.9732481...],
                          [ 700.        ,    0.9806562...],
                          [ 710.        ,    0.9855873...],
                          [ 720.        ,    0.9889903...],
                          [ 730.        ,    0.9914117...],
                          [ 740.        ,    0.9931801...],
                          [ 750.        ,    0.9945009...],
                          [ 760.        ,    0.9955066...],
                          [ 770.        ,    0.9962855...],
                          [ 780.        ,    0.9968976...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2066217...,  0.1220128...,  0.0513958...])

    *Mallett and Yuksel (2019)* reflectance recovery:

    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SPECTRAL_SHAPE_sRGB_MALLETT2019)
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd(XYZ, method='Mallett 2019')
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
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

    *Meng (2015)* reflectance recovery:

    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd(
    ...     XYZ, method='Meng 2015', cmfs=cmfs, illuminant=illuminant)
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

    *Otsu, Yamamoto and Hachisuka (2018)* reflectance recovery:

    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SPECTRAL_SHAPE_OTSU2018)
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd(
    ...     XYZ, method='Otsu 2018', cmfs=cmfs, illuminant=illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.0601939...],
                          [ 390.        ,    0.0568063...],
                          [ 400.        ,    0.0517429...],
                          [ 410.        ,    0.0495841...],
                          [ 420.        ,    0.0502007...],
                          [ 430.        ,    0.0506489...],
                          [ 440.        ,    0.0510020...],
                          [ 450.        ,    0.0493782...],
                          [ 460.        ,    0.0468046...],
                          [ 470.        ,    0.0437132...],
                          [ 480.        ,    0.0416957...],
                          [ 490.        ,    0.0403783...],
                          [ 500.        ,    0.0405197...],
                          [ 510.        ,    0.0406031...],
                          [ 520.        ,    0.0416912...],
                          [ 530.        ,    0.0430956...],
                          [ 540.        ,    0.0444474...],
                          [ 550.        ,    0.0459336...],
                          [ 560.        ,    0.0507631...],
                          [ 570.        ,    0.0628967...],
                          [ 580.        ,    0.0844661...],
                          [ 590.        ,    0.1334277...],
                          [ 600.        ,    0.2262428...],
                          [ 610.        ,    0.3599330...],
                          [ 620.        ,    0.4885571...],
                          [ 630.        ,    0.5752546...],
                          [ 640.        ,    0.6193023...],
                          [ 650.        ,    0.6450744...],
                          [ 660.        ,    0.6610548...],
                          [ 670.        ,    0.6688673...],
                          [ 680.        ,    0.6795426...],
                          [ 690.        ,    0.6887933...],
                          [ 700.        ,    0.7003469...],
                          [ 710.        ,    0.7084128...],
                          [ 720.        ,    0.7154674...],
                          [ 730.        ,    0.7234334...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2065494...,  0.1219712...,  0.0514002...])

    *Smits (1999)* reflectance recovery:

    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['E'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd(XYZ, method='Smits 1999')
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.0787830...],
                          [ 417.7778    ,    0.0622018...],
                          [ 455.5556    ,    0.0446206...],
                          [ 493.3333    ,    0.0352220...],
                          [ 531.1111    ,    0.0324149...],
                          [ 568.8889    ,    0.0330105...],
                          [ 606.6667    ,    0.3207115...],
                          [ 644.4444    ,    0.3836164...],
                          [ 682.2222    ,    0.3836164...],
                          [ 720.        ,    0.3835649...]],
                         interpolator=LinearInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.1894770...,  0.1126470...,  0.0474420...])
    """

    a = as_float_array(XYZ)
    method = validate_method(method, XYZ_TO_SD_METHODS)

    function = XYZ_TO_SD_METHODS[method]

    if function is RGB_to_sd_Smits1999:
        from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999

        a = XYZ_to_RGB_Smits1999(XYZ)
    elif function is RGB_to_sd_Mallett2019:
        from colour.models import XYZ_to_sRGB

        a = XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)

    return function(a, **filter_kwargs(function, **kwargs))


__all__ += [
    "XYZ_TO_SD_METHODS",
    "XYZ_to_sd",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class recovery(ModuleAPI):
    """Define a class acting like the *recovery* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.4.0
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour.recovery.NodeTree_Otsu2018",
            "colour.recovery.Tree_Otsu2018",
        ],
    ]
}
"""Defines the *colour.recovery* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour.recovery"] = recovery(  # type: ignore[assignment]
        sys.modules["colour.recovery"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
