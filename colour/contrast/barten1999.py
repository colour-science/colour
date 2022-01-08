# -*- coding: utf-8 -*-
"""
Barten (1999) Contrast Sensitivity Function
===========================================

Defines the *Barten (1999)* contrast sensitivity function:

-   :func:`colour.contrast.contrast_sensitivity_function_Barten1999`

References
----------
-   :cite:`Barten1999` : Barten, P. G. (1999). Contrast Sensitivity of the
    Human Eye and Its Effects on Image Quality. SPIE. doi:10.1117/3.353254
-   :cite:`Barten2003` : Barten, P. G. J. (2003). Formula for the contrast
    sensitivity of the human eye. In Y. Miyake & D. R. Rasmussen (Eds.),
    Proceedings of SPIE (Vol. 5294, pp. 231-238). doi:10.1117/12.537476
-   :cite:`Cowan2004` : Cowan, M., Kennel, G., Maier, T., & Walker, B. (2004).
    Contrast Sensitivity Experiment to Determine the Bit Depth for Digital
    Cinema. SMPTE Motion Imaging Journal, 113(9), 281-292. doi:10.5594/j11549
-   :cite:`InternationalTelecommunicationUnion2015` : International
    Telecommunication Union. (2015). Report ITU-R BT.2246-4 - The present
    state of ultra-high definition television BT Series Broadcasting service
    (Vol. 5, pp. 1-92).
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2246-4-2015-PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Optional,
)
from colour.utilities import as_float_array, as_float

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'optical_MTF_Barten1999',
    'pupil_diameter_Barten1999',
    'sigma_Barten1999',
    'retinal_illuminance_Barten1999',
    'maximum_angular_size_Barten1999',
    'contrast_sensitivity_function_Barten1999',
]


def optical_MTF_Barten1999(u: FloatingOrArrayLike,
                           sigma: FloatingOrArrayLike = 0.01
                           ) -> FloatingOrNDArray:
    """
    Returns the optical modulation transfer function (MTF) :math:`M_{opt}` of
    the eye using *Barten (1999)* method.

    Parameters
    ----------
    u
        Spatial frequency :math:`u`, the cycles per degree.
    sigma
        Standard deviation :math:`\\sigma` of the line-spread function
        resulting from the convolution of the different elements of the
        convolution process.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Optical modulation transfer function (MTF) :math:`M_{opt}` of the eye.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> optical_MTF_Barten1999(4, 0.01)  # doctest: +ELLIPSIS
    0.9689107...
    """

    u = as_float_array(u)
    sigma = as_float_array(sigma)

    return as_float(np.exp(-2 * np.pi ** 2 * sigma ** 2 * u ** 2))


def pupil_diameter_Barten1999(
        L: FloatingOrArrayLike,
        X_0: FloatingOrArrayLike = 60,
        Y_0: Optional[FloatingOrArrayLike] = None) -> FloatingOrNDArray:
    """
    Returns the pupil diameter for given luminance and object or stimulus
    angular size using *Barten (1999)* method.

    Parameters
    ----------
    L
        Average luminance :math:`L` in :math:`cd/m^2`.
    X_0
        Angular size of the object :math:`X_0` in degrees in the x direction.
    Y_0
        Angular size of the object :math:`X_0` in degrees in the y direction.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Pupil diameter.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> pupil_diameter_Barten1999(100, 60, 60)  # doctest: +ELLIPSIS
    2.0777571...
    """

    L = as_float_array(L)
    X_0 = as_float_array(X_0)
    Y_0 = X_0 if Y_0 is None else as_float_array(Y_0)

    return as_float(5 - 3 * np.tanh(0.4 * np.log(L * X_0 * Y_0 / 40 ** 2)))


def sigma_Barten1999(sigma_0: FloatingOrArrayLike = 0.5 / 60,
                     C_ab: FloatingOrArrayLike = 0.08 / 60,
                     d: FloatingOrArrayLike = 2.1) -> FloatingOrNDArray:
    """
    Returns the standard deviation :math:`\\sigma` of the line-spread function
    resulting from the convolution of the different elements of the convolution
    process using *Barten (1999)* method.

    The :math:`\\sigma` quantity depends on the pupil diameter :math:`d` of the
    eye lens. For very small pupil diameters, :math:`\\sigma` increases
    inversely proportionally with pupil size because of diffraction, and for
    large pupil diameters, :math:`\\sigma` increases about linearly with pupil
    size because of chromatic aberration and others aberrations.

    Parameters
    ----------
    sigma_0
        Constant :math:`\\sigma_{0}` in degrees.
    C_ab
        Spherical aberration of the eye :math:`C_{ab}` in
        :math:`degrees\\div mm`.
    d
        Pupil diameter :math:`d` in millimeters.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Standard deviation :math:`\\sigma` of the line-spread function
        resulting from the convolution of the different elements of the
        convolution process.

    Warnings
    --------
    This definition expects :math:`\\sigma_{0}` and :math:`C_{ab}` to be given
    in degrees and :math:`degrees\\div mm` respectively. However, in the
    literature, the values for :math:`\\sigma_{0}` and
    :math:`C_{ab}` are usually given in :math:`arc min` and
    :math:`arc min\\div mm` respectively, thus they need to be divided by 60.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> sigma_Barten1999(0.5 / 60, 0.08 / 60, 2.1)  # doctest: +ELLIPSIS
    0.0087911...
    """

    sigma_0 = as_float_array(sigma_0)
    C_ab = as_float_array(C_ab)
    d = as_float_array(d)

    return as_float(np.sqrt(sigma_0 ** 2 + (C_ab * d) ** 2))


def retinal_illuminance_Barten1999(
        L: FloatingOrArrayLike,
        d: FloatingOrArrayLike = 2.1,
        apply_stiles_crawford_effect_correction: Boolean = True):
    """
    Returns the retinal illuminance :math:`E` in Trolands for given average
    luminance :math:`L` and pupil diameter :math:`d` using *Barten (1999)*
    method.

    Parameters
    ----------
    L
        Average luminance :math:`L` in :math:`cd/m^2`.
    d
        Pupil diameter :math:`d` in millimeters.
    apply_stiles_crawford_effect_correction
        Whether to apply the correction for *Stiles-Crawford* effect.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Retinal illuminance :math:`E` in Trolands.

    Notes
    -----
    -   This definition is for use with photopic viewing conditions and thus
        corrects for the Stiles-Crawford effect by default, i.e. directional
        sensitivity of the cone cells with lower response of cone cells
        receiving light from the edge of the pupil.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> retinal_illuminance_Barten1999(100, 2.1)  # doctest: +ELLIPSIS
    330.4115803...
    >>> retinal_illuminance_Barten1999(100, 2.1, False)  # doctest: +ELLIPSIS
    346.3605900...
    """

    d = as_float_array(d)
    L = as_float_array(L)

    E = (np.pi * d ** 2) / 4 * L

    if apply_stiles_crawford_effect_correction:
        E *= (1 - (d / 9.7) ** 2 + (d / 12.4) ** 4)

    return as_float(E)


def maximum_angular_size_Barten1999(
        u: FloatingOrArrayLike,
        X_0: FloatingOrArrayLike = 60,
        X_max: FloatingOrArrayLike = 12,
        N_max: FloatingOrArrayLike = 15) -> FloatingOrNDArray:
    """
    Returns the maximum angular size :math:`X` of the object considered using
    *Barten (1999)* method.

    Parameters
    ----------
    u
        Spatial frequency :math:`u`, the cycles per degree.
    X_0
        Angular size :math:`X_0` in degrees of the object in the x direction.
    X_max
        Maximum angular size :math:`X_{max}` in degrees of the integration
        area in the x direction.
    N_max
        Maximum number of cycles :math:`N_{max}` over which the eye can
        integrate the information.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Maximum angular size :math:`X` of the object considered.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> maximum_angular_size_Barten1999(4)  # doctest: +ELLIPSIS
    3.5729480...
    """

    u = as_float_array(u)
    X_0 = as_float_array(X_0)
    X_max = as_float_array(X_max)
    N_max = as_float_array(N_max)

    return as_float((1 / X_0 ** 2 + 1 / X_max ** 2 + u ** 2 / N_max ** 2)
                    ** -0.5)


def contrast_sensitivity_function_Barten1999(
        u: FloatingOrArrayLike,
        sigma: FloatingOrArrayLike = sigma_Barten1999(0.5 / 60, 0.08 / 60,
                                                      2.1),
        k: FloatingOrArrayLike = 3.0,
        T: FloatingOrArrayLike = 0.1,
        X_0: FloatingOrArrayLike = 60,
        Y_0: Optional[FloatingOrArrayLike] = None,
        X_max: FloatingOrArrayLike = 12,
        Y_max: Optional[FloatingOrArrayLike] = None,
        N_max: FloatingOrArrayLike = 15,
        n: FloatingOrArrayLike = 0.03,
        p: FloatingOrArrayLike = 1.2274 * 10 ** 6,
        E: FloatingOrArrayLike = retinal_illuminance_Barten1999(20, 2.1),
        phi_0: FloatingOrArrayLike = 3 * 10 ** -8,
        u_0: FloatingOrArrayLike = 7) -> FloatingOrNDArray:
    """
    Returns the contrast sensitivity :math:`S` of the human eye according to
    the contrast sensitivity function (CSF) described by *Barten (1999)*.

    Contrast sensitivity is defined as the inverse of the modulation threshold
    of a sinusoidal luminance pattern. The modulation threshold of this pattern
    is generally defined by 50% probability of detection. The contrast
    sensitivity function or CSF gives the contrast sensitivity as a function of
    spatial frequency. In the CSF, the spatial frequency is expressed in
    angular units with respect to the eye. It reaches a maximum between 1 and
    10 cycles per degree with a fall off at higher and lower spatial
    frequencies.

    Parameters
    ----------
    u
        Spatial frequency :math:`u`, the cycles per degree.
    sigma
        Standard deviation :math:`\\sigma` of the line-spread function
        resulting from the convolution of the different elements of the
        convolution process.
    k
        Signal-to-noise (SNR) ratio :math:`k`.
    T
        Integration time :math:`T` in seconds of the eye.
    X_0
        Angular size :math:`X_0` in degrees of the object in the x direction.
    Y_0
        Angular size :math:`Y_0` in degrees of the object in the y direction.
    X_max
        Maximum angular size :math:`X_{max}` in degrees of the integration
        area in the x direction.
    Y_max
        Maximum angular size :math:`Y_{max}` in degrees of the integration
        area in the y direction.
    N_max
        Maximum number of cycles :math:`N_{max}` over which the eye can
        integrate the information.
    n
        Quantum efficiency of the eye :math:`n`.
    p
        Photon conversion factor :math:`p` in
        :math:`photons\\div seconds\\div degrees^2\\div Trolands` that
        depends on the light source.
    E
        Retinal illuminance :math:`E` in Trolands.
    phi_0
        Spectral density :math:`\\phi_0` in :math:`seconds degrees^2` of the
        neural noise.
    u_0
        Spatial frequency :math:`u_0` in :math:`cycles\\div degrees` above
        which the lateral inhibition ceases.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Contrast sensitivity :math:`S`.

    Warnings
    --------
    This definition expects :math:`\\sigma_{0}` and :math:`C_{ab}` used in the
    computation of :math:`\\sigma` to be given in degrees and
    :math:`degrees\\div mm` respectively. However, in the literature, the
    values for :math:`\\sigma_{0}` and :math:`C_{ab}` are usually given in
    :math:`arc min` and :math:`arc min\\div mm` respectively, thus they need to
    be divided by 60.

    Notes
    -----
    -   The formula holds for bilateral viewing and for equal dimensions of
        the object in x and y direction. For monocular vision, the contrast
        sensitivity is a factor :math:`\\sqrt{2}` smaller.
    -   *Barten (1999)* CSF default values for the :math:`k`,
        :math:`\\sigma_{0}`, :math:`C_{ab}`, :math:`T`, :math:`X_{max}`,
        :math:`N_{max}`, :math:`n`, :math:`\\phi_{0}` and :math:`u_0` constants
        are valid for a standard observer with good vision and with an age
        between 20 and 30 years.
    -   The other constants have been filled using reference data from
        *Figure 31* in :cite:`InternationalTelecommunicationUnion2015` but
        must be adapted to the current use case.
    -   The product of :math:`u`, the cycles per degree, and :math:`X_0`,
        the number of degrees, gives the number of cycles :math:`P_c` in a
        pattern. Therefore, :math:`X_0` can be made a variable dependent on
        :math:`u` such as :math:`X_0 = P_c / u`.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> contrast_sensitivity_function_Barten1999(4)  # doctest: +ELLIPSIS
    360.8691122...

    Reproducing *Figure 31* in \
:cite:`InternationalTelecommunicationUnion2015` illustrating the minimum
    detectable contrast according to *Barten (1999)* model with the assumed
    conditions for UHDTV applications. The minimum detectable contrast
    :math:`MDC` is then defined as follows::

        :math:`MDC = 1 / CSF * 2 * (1 / 1.27)`

    where :math:`2` is used for the conversion from modulation to contrast and
    :math:`1 / 1.27` is used for the conversion from sinusoidal to rectangular
    waves.

    >>> from scipy.optimize import fmin
    >>> settings_BT2246 = {
    ...     'k': 3.0,
    ...     'T': 0.1,
    ...     'X_max': 12,
    ...     'N_max': 15,
    ...     'n': 0.03,
    ...     'p': 1.2274 * 10 ** 6,
    ...     'phi_0': 3 * 10 ** -8,
    ...     'u_0': 7,
    ... }
    >>>
    >>> def maximise_spatial_frequency(L):
    ...     maximised_spatial_frequency = []
    ...     for L_v in L:
    ...         X_0 = 60
    ...         d = pupil_diameter_Barten1999(L_v, X_0)
    ...         sigma = sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
    ...         E = retinal_illuminance_Barten1999(L_v, d, True)
    ...         maximised_spatial_frequency.append(
    ...             fmin(lambda x: (
    ...                     -contrast_sensitivity_function_Barten1999(
    ...                         u=x,
    ...                         sigma=sigma,
    ...                         X_0=X_0,
    ...                         E=E,
    ...                         **settings_BT2246)
    ...                 ), 0, disp=False)[0])
    ...     return as_float(np.array(maximised_spatial_frequency))
    >>>
    >>> L = np.logspace(np.log10(0.01), np.log10(100), 10)
    >>> X_0 = Y_0 = 60
    >>> d = pupil_diameter_Barten1999(L, X_0, Y_0)
    >>> sigma = sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
    >>> E = retinal_illuminance_Barten1999(L, d)
    >>> u = maximise_spatial_frequency(L)
    >>> (1 / contrast_sensitivity_function_Barten1999(
    ...     u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246)
    ...  * 2 * (1/ 1.27))
    ... # doctest: +ELLIPSIS
    array([ 0.0207396...,  0.0133019...,  0.0089256...,  0.0064202...,  \
0.0050275...,
            0.0041933...,  0.0035573...,  0.0030095...,  0.0025803...,  \
0.0022897...])
    """

    u = as_float_array(u)
    k = as_float_array(k)
    T = as_float_array(T)
    X_0 = as_float_array(X_0)
    Y_0 = X_0 if Y_0 is None else as_float_array(Y_0)
    X_max = as_float_array(X_max)
    Y_max = X_max if Y_max is None else as_float_array(Y_max)
    N_max = as_float_array(N_max)
    n = as_float_array(n)
    p = as_float_array(p)
    E = as_float_array(E)
    phi_0 = as_float_array(phi_0)
    u_0 = as_float_array(u_0)

    M_opt = optical_MTF_Barten1999(u, sigma)

    M_as = 1 / (maximum_angular_size_Barten1999(u, X_0, X_max, N_max) *
                maximum_angular_size_Barten1999(u, Y_0, Y_max, N_max))

    S = (M_opt / k) / np.sqrt(2 / T * M_as * (1 / (n * p * E) + phi_0 /
                                              (1 - np.exp(-(u / u_0) ** 2))))

    return as_float(S)
