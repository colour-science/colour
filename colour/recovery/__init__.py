# -*- coding: utf-8 -*-
"""
References
----------
-   :cite:`Meng2015c` : Meng, J., Simon, F., Hanika, J., & Dachsbacher, C.
    (2015). Physically Meaningful Rendering using Tristimulus Colours. Computer
    Graphics Forum, 34(4), 31-40. doi:10.1111/cgf.12676
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147–155. doi:10.1111/cgf.13626
"""

from __future__ import absolute_import

from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              filter_kwargs)

from .datasets import *  # noqa
from . import datasets
from .meng2015 import XYZ_to_sd_Meng2015
from .smits1999 import RGB_to_sd_Smits1999
from .jakob2019 import (sd_Jakob2019, find_coefficients_Jakob2019,
                        XYZ_to_sd_Jakob2019, Jakob2019Interpolator)
__all__ = []
__all__ += datasets.__all__
__all__ += ['XYZ_to_sd_Meng2015']
__all__ += ['RGB_to_sd_Smits1999']
__all__ += [
    'sd_Jakob2019', 'find_coefficients_Jakob2019', 'XYZ_to_sd_Jakob2019',
    'Jakob2019Interpolator'
]

XYZ_TO_SD_METHODS = CaseInsensitiveMapping({
    'Jakob 2019': XYZ_to_sd_Jakob2019,
    'Meng 2015': XYZ_to_sd_Meng2015,
    'Smits 1999': RGB_to_sd_Smits1999,
})
XYZ_TO_SD_METHODS.__doc__ = """
Supported spectral distribution recovery methods.

References
----------
:cite:`Jakob2019Spectral`, :cite:`Meng2015c`, :cite:`Smits1999a`

XYZ_TO_SD_METHODS : CaseInsensitiveMapping
    **{'Jakob 2019', 'Meng 2015', 'Smits 1999'}**
"""


def XYZ_to_sd(XYZ, method='Meng 2015', **kwargs):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus
    values using given method.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values to recover the spectral distribution
        from.
    method : unicode, optional
        **{'Meng 2015', 'Jakob 2019', 'Smits 1999'}**
        Computation method.

    Other Parameters
    ----------------
    additional_data : bool, optional
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`},
        If *True*, ``error`` will be returned alongside ``sd``.
    cmfs : XYZ_ColourMatchingFunctions, optional
        {:func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Standard observer colour matching functions.
    colourspace : RGB_Colourspace, optional
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`},
        *RGB* colourspace of the target colour. Note that no chromatic
        adaptation is performed between ``illuminant`` and the colourspace
        whitepoint.
    illuminant : SpectralDistribution, optional
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`,
        :func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Illuminant spectral distribution.
    interval : numeric, optional
        {:func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Wavelength :math:`\\lambda_{i}` range interval in nm. The smaller
        ``interval`` is, the longer the computations will be.
    optimisation_kwargs : dict_like, optional
        {:func:`colour.recovery.XYZ_to_sd_Jakob2019`,
        :func:`colour.recovery.XYZ_to_sd_Meng2015`},
        Parameters for :func:`scipy.optimize.minimize` and
        :func:`colour.recovery.find_coefficients_Jakob2019` definitions.

    Returns
    -------
    SpectralDistribution
        Recovered spectral distribution.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   *Smits (1999)* method will internally convert given *CIE XYZ*
        tristimulus values to *RGB* colourspace array assuming equal energy
        illuminant *E*.

    References
    ----------
    :cite:`Jakob2019Spectral`, :cite:`Meng2015c`, :cite:`Smits1999a`

    Examples
    --------

    *Meng (2015)* reflectance recovery:

    >>> import numpy as np
    >>> from colour.utilities import numpy_print_options
    >>> from colour.colorimetry import (
    ...     MSDS_CMFS_STANDARD_OBSERVER, SpectralShape, sd_to_XYZ_integration)
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> sd = XYZ_to_sd(XYZ, cmfs=cmfs)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.0780114...],
                          [ 370.        ,    0.0780316...],
                          [ 380.        ,    0.0780471...],
                          [ 390.        ,    0.0780351...],
                          [ 400.        ,    0.0779702...],
                          [ 410.        ,    0.0778033...],
                          [ 420.        ,    0.0770958...],
                          [ 430.        ,    0.0748008...],
                          [ 440.        ,    0.0693230...],
                          [ 450.        ,    0.0601136...],
                          [ 460.        ,    0.0477407...],
                          [ 470.        ,    0.0334964...],
                          [ 480.        ,    0.0193352...],
                          [ 490.        ,    0.0074858...],
                          [ 500.        ,    0.0001225...],
                          [ 510.        ,    0.       ...],
                          [ 520.        ,    0.       ...],
                          [ 530.        ,    0.       ...],
                          [ 540.        ,    0.0124896...],
                          [ 550.        ,    0.0389831...],
                          [ 560.        ,    0.0775105...],
                          [ 570.        ,    0.1247947...],
                          [ 580.        ,    0.1765339...],
                          [ 590.        ,    0.2281918...],
                          [ 600.        ,    0.2751347...],
                          [ 610.        ,    0.3140115...],
                          [ 620.        ,    0.3433561...],
                          [ 630.        ,    0.3635777...],
                          [ 640.        ,    0.3765428...],
                          [ 650.        ,    0.3841726...],
                          [ 660.        ,    0.3883633...],
                          [ 670.        ,    0.3905415...],
                          [ 680.        ,    0.3916742...],
                          [ 690.        ,    0.3922554...],
                          [ 700.        ,    0.3925427...],
                          [ 710.        ,    0.3926783...],
                          [ 720.        ,    0.3927330...],
                          [ 730.        ,    0.3927586...],
                          [ 740.        ,    0.3927548...],
                          [ 750.        ,    0.3927681...],
                          [ 760.        ,    0.3927813...],
                          [ 770.        ,    0.3927840...],
                          [ 780.        ,    0.3927536...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.2178545...,  0.1254141...,  0.0470095...])

    *Jakob and Hanika (2009)* reflectance recovery:

    >>> sd = XYZ_to_sd(XYZ, cmfs=cmfs, method='Jakob 2019')
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.3692754...],
                          [ 370.        ,    0.2470157...],
                          [ 380.        ,    0.1702836...],
                          [ 390.        ,    0.1237443...],
                          [ 400.        ,    0.0948289...],
                          [ 410.        ,    0.0761417...],
                          [ 420.        ,    0.0636055...],
                          [ 430.        ,    0.0549472...],
                          [ 440.        ,    0.0488572...],
                          [ 450.        ,    0.0445552...],
                          [ 460.        ,    0.0415635...],
                          [ 470.        ,    0.0395874...],
                          [ 480.        ,    0.0384489...],
                          [ 490.        ,    0.038052 ...],
                          [ 500.        ,    0.0383639...],
                          [ 510.        ,    0.0394104...],
                          [ 520.        ,    0.0412793...],
                          [ 530.        ,    0.0441372...],
                          [ 540.        ,    0.0482625...],
                          [ 550.        ,    0.0541060...],
                          [ 560.        ,    0.0624031...],
                          [ 570.        ,    0.0743826...],
                          [ 580.        ,    0.0921694...],
                          [ 590.        ,    0.1195616...],
                          [ 600.        ,    0.1634560...],
                          [ 610.        ,    0.2357564...],
                          [ 620.        ,    0.3520400...],
                          [ 630.        ,    0.5140310...],
                          [ 640.        ,    0.6821088...],
                          [ 650.        ,    0.8073255...],
                          [ 660.        ,    0.8831221...],
                          [ 670.        ,    0.9262541...],
                          [ 680.        ,    0.9512104...],
                          [ 690.        ,    0.9662805...],
                          [ 700.        ,    0.9758111...],
                          [ 710.        ,    0.9820992...],
                          [ 720.        ,    0.9864037...],
                          [ 730.        ,    0.9894449...],
                          [ 740.        ,    0.9916522...],
                          [ 750.        ,    0.9932920...],
                          [ 760.        ,    0.9945349...],
                          [ 770.        ,    0.9954934...],
                          [ 780.        ,    0.9962442...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.2177098...,  0.1253832...,  0.0469567...])

    *Smits (1999)* reflectance recovery:

    >>> sd = XYZ_to_sd(XYZ, method='Smits 1999')
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.07691923],
                          [ 417.7778    ,    0.0587005 ],
                          [ 455.5556    ,    0.03943195],
                          [ 493.3333    ,    0.03024978],
                          [ 531.1111    ,    0.02750692],
                          [ 568.8889    ,    0.02808645],
                          [ 606.6667    ,    0.34298985],
                          [ 644.4444    ,    0.41185795],
                          [ 682.2222    ,    0.41185795],
                          [ 720.        ,    0.41180754]],
                         interpolator=LinearInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.1996032...,  0.1155770...,  0.0427866...])
    """

    a = as_float_array(XYZ)

    function = XYZ_TO_SD_METHODS[method]

    if function is RGB_to_sd_Smits1999:
        from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999

        a = XYZ_to_RGB_Smits1999(XYZ)

    return function(a, **filter_kwargs(function, **kwargs))


__all__ += ['XYZ_TO_SD_METHODS', 'XYZ_to_sd']
