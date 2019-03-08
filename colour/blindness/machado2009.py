# -*- coding: utf-8 -*-
"""
Simulation of CVD - Machado, Oliveira and Fernandes (2009)
==========================================================

Defines *Machado et al. (2009)* objects for simulation of colour vision
deficiency:

-   :func:`colour.anomalous_trichromacy_cmfs_Machado2009`
-   :func:`colour.anomalous_trichromacy_matrix_Machado2009`
-   :func:`colour.cvd_matrix_Machado2009`

See Also
--------
`Machado et al. (2009) - CVD IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/cvd/Machado2009.ipynb>`_

References
----------
-   :cite:`Colblindorb` : Colblindor. (n.d.). Protanopia - Red-Green Color
    Blindness. Retrieved July 4, 2015, from http://www.color-blindness.com/\
protanopia-red-green-color-blindness/
-   :cite:`Colblindora` : Colblindor. (n.d.). Deuteranopia - Red-Green Color
    Blindness. Retrieved July 4, 2015, from http://www.color-blindness.com/\
deuteranopia-red-green-color-blindness/
-   :cite:`Colblindorc` : Colblindor. (n.d.). Tritanopia - Blue-Yellow Color
    Blindness. Retrieved July 4, 2015, from http://www.color-blindness.com/\
tritanopia-blue-yellow-color-blindness/
-   :cite:`Machado2009` : Machado, G. M., Oliveira, M. M., & Fernandes, L.
    (2009). A Physiologically-based Model for Simulation of Color Vision
    Deficiency. IEEE Transactions on Visualization and Computer Graphics,
    15(6), 1291-1298. doi:10.1109/TVCG.2009.113
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.blindness import CVD_MATRICES_MACHADO2010
from colour.colorimetry import SpectralShape
from colour.utilities import (dot_matrix, dot_vector, tsplit, tstack,
                              usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'LMS_TO_WSYBRG_MATRIX', 'RGB_to_WSYBRG_matrix',
    'anomalous_trichromacy_cmfs_Machado2009',
    'anomalous_trichromacy_matrix_Machado2009', 'cvd_matrix_Machado2009'
]

LMS_TO_WSYBRG_MATRIX = np.array([
    [0.600, 0.400, 0.000],
    [0.240, 0.105, -0.700],
    [1.200, -1.600, 0.400],
])
"""
Ingling and Tsou (1977) matrix converting from cones responses to
opponent-colour space.

LMS_TO_WSYBRG_MATRIX : array_like, (3, 3)
"""


def RGB_to_WSYBRG_matrix(cmfs, primaries):
    """
    Computes the matrix transforming from *RGB* colourspace to opponent-colour
    space using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs : LMS_ConeFundamentals
        *LMS* cone fundamentals colour matching functions.
    primaries : RGB_DisplayPrimaries
        *RGB* display primaries tri-spectral distributions.

    Returns
    -------
    ndarray
        Matrix transforming from *RGB* colourspace to opponent-colour space.

    Examples
    --------
    >>> from colour import DISPLAYS_RGB_PRIMARIES, LMS_CMFS
    >>> cmfs = LMS_CMFS['Stockman & Sharpe 2 Degree Cone Fundamentals']
    >>> d_LMS = np.array([15, 0, 0])
    >>> primaries = DISPLAYS_RGB_PRIMARIES['Apple Studio Display']
    >>> RGB_to_WSYBRG_matrix(  # doctest: +ELLIPSIS
    ...     cmfs, primaries)
    array([[  0.2126535...,   0.6704626...,   0.1168838...],
           [  4.7095295...,  12.4862869..., -16.1958165...],
           [-11.1518474...,  15.2534789...,  -3.1016315...]])
    """

    wavelengths = cmfs.wavelengths
    WSYBRG = dot_vector(LMS_TO_WSYBRG_MATRIX, cmfs.values)
    WS, YB, RG = tsplit(WSYBRG)

    extrapolator_args = {'method': 'Constant', 'left': 0, 'right': 0}
    primaries = primaries.copy().align(
        cmfs.shape, extrapolator_args=extrapolator_args)

    R, G, B = tsplit(primaries.values)

    WS_R = np.trapz(R * WS, wavelengths)
    WS_G = np.trapz(G * WS, wavelengths)
    WS_B = np.trapz(B * WS, wavelengths)

    YB_R = np.trapz(R * YB, wavelengths)
    YB_G = np.trapz(G * YB, wavelengths)
    YB_B = np.trapz(B * YB, wavelengths)

    RG_R = np.trapz(R * RG, wavelengths)
    RG_G = np.trapz(G * RG, wavelengths)
    RG_B = np.trapz(B * RG, wavelengths)

    M_G = np.array([
        [WS_R, WS_G, WS_B],
        [YB_R, YB_G, YB_B],
        [RG_R, RG_G, RG_B],
    ])

    PWS = 1 / (WS_R + WS_G + WS_B)
    PYB = 1 / (YB_R + YB_G + YB_B)
    PRG = 1 / (RG_R + RG_G + RG_B)

    M_G *= np.array([PWS, PYB, PRG])[:, np.newaxis]

    return M_G


def anomalous_trichromacy_cmfs_Machado2009(cmfs, d_LMS):
    """
    Shifts given *LMS* cone fundamentals colour matching functions with given
    :math:`\\Delta_{LMS}` shift amount in nanometers to simulate anomalous
    trichromacy using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs : LMS_ConeFundamentals
        *LMS* cone fundamentals colour matching functions.
    d_LMS : array_like
        :math:`\\Delta_{LMS}` shift amount in nanometers.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions interval is
        expected to be 1 nanometer, incompatible input will be interpolated
        at 1 nanometer interval.
    -   Input :math:`\\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    LMS_ConeFundamentals
        Anomalous trichromacy *LMS* cone fundamentals colour matching
        functions.

    Warning
    -------
    *Machado et al. (2009)* simulation of tritanomaly is based on the shift
    paradigm as an approximation to the actual phenomenon and restrain the
    model from trying to model tritanopia.
    The pre-generated matrices are using a shift value in domain [5, 59]
    contrary to the domain [0, 20] used for protanomaly and deuteranomaly
    simulation.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> from colour import LMS_CMFS
    >>> cmfs = LMS_CMFS['Stockman & Sharpe 2 Degree Cone Fundamentals']
    >>> cmfs[450]
    array([ 0.0498639,  0.0870524,  0.955393 ])
    >>> anomalous_trichromacy_cmfs_Machado2009(cmfs, np.array([15, 0, 0]))[450]
    ... # doctest: +ELLIPSIS
    array([ 0.0891288...,  0.0870524 ,  0.955393  ])
    """

    cmfs = cmfs.copy()
    if cmfs.shape.interval != 1:
        cmfs.interpolate(SpectralShape(interval=1))

    cmfs.extrapolator_args = {'method': 'Constant', 'left': 0, 'right': 0}

    L, M, _S = tsplit(cmfs.values)
    d_L, d_M, d_S = tsplit(d_LMS)

    if d_S != 0:
        usage_warning(
            '"Machado et al. (2009)" simulation of tritanomaly is based on '
            'the shift paradigm as an approximation to the actual phenomenon '
            'and restrain the model from trying to model tritanopia.\n'
            'The pre-generated matrices are using a shift value in domain '
            '[5, 59] contrary to the domain [0, 20] used for protanomaly and '
            'deuteranomaly simulation.')

    area_L = np.trapz(L, cmfs.wavelengths)
    area_M = np.trapz(M, cmfs.wavelengths)

    def alpha(x):
        """
        Computes :math:`alpha` factor.
        """

        return (20 - x) / 20

    # Corrected equations as per:
    # http://www.inf.ufrgs.br/~oliveira/pubs_files/
    # CVD_Simulation/CVD_Simulation.html#Errata
    L_a = alpha(d_L) * L + 0.96 * area_L / area_M * (1 - alpha(d_L)) * M
    M_a = alpha(d_M) * M + 1 / 0.96 * area_M / area_L * (1 - alpha(d_M)) * L
    S_a = cmfs[cmfs.wavelengths - d_S][:, 2]

    LMS_a = tstack([L_a, M_a, S_a])
    cmfs[cmfs.wavelengths] = LMS_a

    severity = '{0}, {1}, {2}'.format(d_L, d_M, d_S)
    template = '{0} - Anomalous Trichromacy ({1})'
    cmfs.name = template.format(cmfs.name, severity)
    cmfs.strict_name = template.format(cmfs.strict_name, severity)

    return cmfs


def anomalous_trichromacy_matrix_Machado2009(cmfs, primaries, d_LMS):
    """
    Computes *Machado et al. (2009)* *CVD* matrix for given *LMS* cone
    fundamentals colour matching functions and display primaries tri-spectral
    distributions with given :math:`\\Delta_{LMS}` shift amount in nanometers
    to simulate anomalous trichromacy.

    Parameters
    ----------
    cmfs : LMS_ConeFundamentals
        *LMS* cone fundamentals colour matching functions.
    primaries : RGB_DisplayPrimaries
        *RGB* display primaries tri-spectral distributions.
    d_LMS : array_like
        :math:`\\Delta_{LMS}` shift amount in nanometers.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions interval is
        expected to be 1 nanometer, incompatible input will be interpolated
        at 1 nanometer interval.
    -   Input :math:`\\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    ndarray
        Anomalous trichromacy matrix.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> from colour import DISPLAYS_RGB_PRIMARIES, LMS_CMFS
    >>> cmfs = LMS_CMFS['Stockman & Sharpe 2 Degree Cone Fundamentals']
    >>> d_LMS = np.array([15, 0, 0])
    >>> primaries = DISPLAYS_RGB_PRIMARIES['Apple Studio Display']
    >>> anomalous_trichromacy_matrix_Machado2009(cmfs, primaries, d_LMS)
    ... # doctest: +ELLIPSIS
    array([[-0.2777465...,  2.6515008..., -1.3737543...],
           [ 0.2718936...,  0.2004786...,  0.5276276...],
           [ 0.0064404...,  0.2592157...,  0.7343437...]])
    """

    if cmfs.shape.interval != 1:
        cmfs = cmfs.copy().interpolate(SpectralShape(interval=1))

    M_n = RGB_to_WSYBRG_matrix(cmfs, primaries)
    cmfs_a = anomalous_trichromacy_cmfs_Machado2009(cmfs, d_LMS)
    M_a = RGB_to_WSYBRG_matrix(cmfs_a, primaries)

    return dot_matrix(np.linalg.inv(M_n), M_a)


def cvd_matrix_Machado2009(deficiency, severity):
    """
    Computes *Machado et al. (2009)* *CVD* matrix for given deficiency and
    severity using the pre-computed matrices dataset.

    Parameters
    ----------
    deficiency : unicode
        {'Protanomaly', 'Deuteranomaly', 'Tritanomaly'}
        Colour blindness / vision deficiency types :
        - *Protanomaly* : defective long-wavelength cones (L-cones). The
        complete absence of L-cones is known as *Protanopia* or
        *red-dichromacy*.
        - *Deuteranomaly* : defective medium-wavelength cones (M-cones) with
        peak of sensitivity moved towards the red sensitive cones. The complete
        absence of M-cones is known as *Deuteranopia*.
        - *Tritanomaly* : defective short-wavelength cones (S-cones), an
        alleviated form of blue-yellow color blindness. The complete absence of
        S-cones is known as *Tritanopia*.
    severity : numeric
        Severity of the colour vision deficiency in domain [0, 1].

    Returns
    -------
    ndarray
        *CVD* matrix.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> cvd_matrix_Machado2009('Protanomaly', 0.15)  # doctest: +ELLIPSIS
    array([[ 0.7869875...,  0.2694875..., -0.0564735...],
           [ 0.0431695...,  0.933774 ...,  0.023058 ...],
           [-0.004238 ..., -0.0024515...,  1.0066895...]])
    """

    if deficiency.lower() == 'tritanomaly':
        usage_warning(
            '"Machado et al. (2009)" simulation of tritanomaly is based on '
            'the shift paradigm as an approximation to the actual phenomenon '
            'and restrain the model from trying to model tritanopia.\n'
            'The pre-generated matrices are using a shift value in domain '
            '[5, 59] contrary to the domain [0, 20] used for protanomaly and '
            'deuteranomaly simulation.')

    matrices = CVD_MATRICES_MACHADO2010[deficiency]
    samples = np.array(sorted(matrices.keys()))
    index = min(np.searchsorted(samples, severity), len(samples) - 1)

    a = samples[index]
    b = samples[min(index + 1, len(samples) - 1)]

    m1, m2 = matrices[a], matrices[b]

    if a == b:
        # 1.0 severity CVD matrix, returning directly.
        return m1
    else:
        return m1 + (severity - a) * ((m2 - m1) / (b - a))
