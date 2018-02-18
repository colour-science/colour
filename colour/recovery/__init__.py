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
"""

from __future__ import absolute_import

import numpy as np

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .dataset import *  # noqa
from . import dataset
from .meng2015 import XYZ_to_spectral_Meng2015
from .smits1999 import RGB_to_spectral_Smits1999

__all__ = []
__all__ += dataset.__all__
__all__ += ['XYZ_to_spectral_Meng2015']
__all__ += ['RGB_to_spectral_Smits1999']

REFLECTANCE_RECOVERY_METHODS = CaseInsensitiveMapping({
    'Meng 2015': XYZ_to_spectral_Meng2015,
    'Smits 1999': RGB_to_spectral_Smits1999,
})
REFLECTANCE_RECOVERY_METHODS.__doc__ = """
Supported reflectance recovery methods.

References
----------
-   :cite:`Meng2015c`
-   :cite:`Smits1999a`

REFLECTANCE_RECOVERY_METHODS : CaseInsensitiveMapping
    **{'Meng 2015', 'Smits 1999'}**
"""


def XYZ_to_spectral(XYZ, method='Meng 2015', **kwargs):
    """
    Recovers the spectral power distribution of given *CIE XYZ* tristimulus
    values using given method.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values to recover the spectral power distribution
        from.
    method : unicode, optional
        **{'Meng 2015', 'Smits 1999'}**,
        Computation method.

    Other Parameters
    ----------------
    cmfs : XYZ_ColourMatchingFunctions
        {:func:`colour.recovery.XYZ_to_spectral_Meng2015`},
        Standard observer colour matching functions.
    interval : numeric, optional
        {:func:`colour.recovery.XYZ_to_spectral_Meng2015`},
        Wavelength :math:`\lambda_{i}` range interval in nm. The smaller
        ``interval`` is, the longer the computations will be.
    tolerance : numeric, optional
        {:func:`colour.recovery.XYZ_to_spectral_Meng2015`},
        Tolerance for termination. The lower ``tolerance`` is, the smoother
        the recovered spectral power distribution will be.
    maximum_iterations : int, optional
        {:func:`colour.recovery.XYZ_to_spectral_Meng2015`},
        Maximum number of iterations to perform.

    Returns
    -------
    SpectralPowerDistribution
        Recovered spectral power distribution.

    Notes
    -----
    -   *Smits (1999)* method will internally convert given *CIE XYZ*
        tristimulus values to *RGB* colourspace array assuming equal energy
        illuminant *E*.

    References
    ----------
    -   :cite:`Meng2015c`
    -   :cite:`Smits1999a`

    Examples
    --------

    *Meng (2015)* reflectance recovery:

    >>> from colour.utilities import numpy_print_options
    >>> from colour.colorimetry import spectral_to_XYZ_integration
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> spd = XYZ_to_spectral(XYZ, interval=10)
    >>> with numpy_print_options(suppress=True):
    ...     spd  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 360.        ,    0.0788075...],
                               [ 370.        ,    0.0788543...],
                               [ 380.        ,    0.0788825...],
                               [ 390.        ,    0.0788714...],
                               [ 400.        ,    0.0788911...],
                               [ 410.        ,    0.07893  ...],
                               [ 420.        ,    0.0797471...],
                               [ 430.        ,    0.0813339...],
                               [ 440.        ,    0.0840145...],
                               [ 450.        ,    0.0892826...],
                               [ 460.        ,    0.0965359...],
                               [ 470.        ,    0.1053176...],
                               [ 480.        ,    0.1150921...],
                               [ 490.        ,    0.1244252...],
                               [ 500.        ,    0.1326083...],
                               [ 510.        ,    0.1390282...],
                               [ 520.        ,    0.1423548...],
                               [ 530.        ,    0.1414636...],
                               [ 540.        ,    0.1365195...],
                               [ 550.        ,    0.1277319...],
                               [ 560.        ,    0.1152622...],
                               [ 570.        ,    0.1004513...],
                               [ 580.        ,    0.0844187...],
                               [ 590.        ,    0.0686863...],
                               [ 600.        ,    0.0543013...],
                               [ 610.        ,    0.0423486...],
                               [ 620.        ,    0.0333861...],
                               [ 630.        ,    0.0273558...],
                               [ 640.        ,    0.0233407...],
                               [ 650.        ,    0.0211208...],
                               [ 660.        ,    0.0197248...],
                               [ 670.        ,    0.0187157...],
                               [ 680.        ,    0.0181510...],
                               [ 690.        ,    0.0179691...],
                               [ 700.        ,    0.0179247...],
                               [ 710.        ,    0.0178665...],
                               [ 720.        ,    0.0178005...],
                               [ 730.        ,    0.0177570...],
                               [ 740.        ,    0.0177090...],
                               [ 750.        ,    0.0175743...],
                               [ 760.        ,    0.0175058...],
                               [ 770.        ,    0.0174492...],
                               [ 780.        ,    0.0174984...],
                               [ 790.        ,    0.0175667...],
                               [ 800.        ,    0.0175657...],
                               [ 810.        ,    0.0175319...],
                               [ 820.        ,    0.0175184...],
                               [ 830.        ,    0.0175390...]],
                              interpolator=SpragueInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
    >>> spectral_to_XYZ_integration(spd) / 100  # doctest: +ELLIPSIS
    array([ 0.0705100...,  0.1007987...,  0.0956738...])

    *Smits (1999)* reflectance recovery:

    >>> spd = XYZ_to_spectral(XYZ, method='Smits 1999')
    >>> with numpy_print_options(suppress=True):
    ...     spd  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 380.        ,    0.0908046...],
                               [ 417.7778    ,    0.0887761...],
                               [ 455.5556    ,    0.0939795...],
                               [ 493.3333    ,    0.1236033...],
                               [ 531.1111    ,    0.1315788...],
                               [ 568.8889    ,    0.1293411...],
                               [ 606.6667    ,    0.0392680...],
                               [ 644.4444    ,    0.0214496...],
                               [ 682.2222    ,    0.0214496...],
                               [ 720.        ,    0.0215462...]],
                              interpolator=CubicSplineInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
    >>> spectral_to_XYZ_integration(spd) / 100  # doctest: +ELLIPSIS
    array([ 0.0753341...,  0.1054586...,  0.0977855...])
    """

    a = np.asarray(XYZ)

    function = REFLECTANCE_RECOVERY_METHODS[method]

    if function is RGB_to_spectral_Smits1999:
        from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999

        a = XYZ_to_RGB_Smits1999(XYZ)

    return function(a, **filter_kwargs(function, **kwargs))


__all__ += ['REFLECTANCE_RECOVERY_METHODS', 'XYZ_to_spectral']
