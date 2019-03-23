# -*- coding: utf-8 -*-
"""
References
----------
-   :cite:`Barten1999` : Barten, P. G. (1999). Contrast Sensitivity of the
    Human Eye and Its Effects on Image Quality. SPIE. doi:10.1117/3.353254
-   :cite:`Barten2003` : Barten, P. G. J. (2003). Formula for the contrast
    sensitivity of the human eye. In Y. Miyake & D. R. Rasmussen (Eds.),
    Proceedings of SPIE (Vol. 5294, pp. 231-238). doi:10.1117/12.537476
-   :cite:`Cowan2004` : Cowan, M., Kennel, G., Maier, T., Walker, B., & Cowan,
    M. (2004). Constant Sensitivity Experiment to Determine the Bit Depth for
    Digital Cinema, (September). Retrieved from
    http://car.france3.mars.free.fr/Formation INA HD/HDTV/HDTV  2007 v35/\
SMPTE normes et confs/Contrastm.pdf
-   :cite:`InternationalTelecommunicationUnion2015` : International
    Telecommunication Union. (2015). Report ITU-R BT.2246-4 - The present state
    of ultra-high definition television BT Series Broadcasting service, 5,
    1-92.
"""

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs
from .barten1999 import (optical_MTF_Barten1999, pupil_diameter_Barten1999,
                         sigma_Barten1999, retinal_illuminance_Barten1999,
                         maximum_angular_size_Barten1999,
                         contrast_sensitivity_function_Barten1999)

__all__ = [
    'optical_MTF_Barten1999', 'pupil_diameter_Barten1999', 'sigma_Barten1999',
    'retinal_illuminance_Barten1999', 'maximum_angular_size_Barten1999',
    'contrast_sensitivity_function_Barten1999'
]

CONTRAST_SENSITIVITY_METHODS = CaseInsensitiveMapping({
    'Barten 1999': contrast_sensitivity_function_Barten1999,
})
CONTRAST_SENSITIVITY_METHODS.__doc__ = """
Supported contrast sensitivity methods.

References
----------
:cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
:cite:`InternationalTelecommunicationUnion2015`,

CONTRAST_SENSITIVITY_METHODS : CaseInsensitiveMapping
    **{'Barten 1999'}**
"""


def contrast_sensitivity_function(method='Barten 1999', **kwargs):
    """
    Returns the contrast sensitivity :math:`S` of the human eye according to
    the contrast sensitivity function (CSF) described by given method.

    Parameters
    ----------
    method : unicode, optional
        **{'Barten 1999'}**,
        Computation method.

    Other Parameters
    ----------------
    E : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Retinal illuminance :math:`E` in Trolands.
    N_max : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Maximum number of cycles :math:`N_{max}` over which the eye can
        integrate the information.
    T : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Integration time :math:`T` in seconds of the eye.
    X_0 : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Angular size :math:`X_0` in degrees of the object in the x direction.
    Y_0 : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Angular size :math:`Y_0` in degrees of the object in the y direction.
    X_max : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Maximum angular size :math:`X_{max}` in degrees of the integration
        area in the x direction.
    Y_max : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Maximum angular size :math:`Y_{max}` in degrees of the integration
        area in the y direction.
    k : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Signal-to-noise (SNR) ratio :math:`k`.
    n : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Quantum efficiency of the eye :math:`n`.
    p : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Photon conversion factor :math:`p` in
        :math:`photons\\div seconds\\div degrees^2\\div Trolands` that
        depends on the light source.
    phi_0 : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Spectral density :math:`\\phi_0` in :math:`seconds degrees^2` of the
        neural noise.
    sigma : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Standard deviation :math:`\\sigma` of the line-spread function
        resulting from the convolution of the different elements of the
        convolution process.
    u : numeric
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Spatial frequency :math:`u`, the cycles per degree.
    u_0 : numeric or array_like, optional
        {:func:`colour.contrast.contrast_sensitivity_function_Barten1999`},
        Spatial frequency :math:`u_0` in :math:`cycles\\div degrees` above
        which the lateral inhibition ceases.

    Returns
    -------
    ndarray
        Contrast sensitivity :math:`S`.

    References
    ----------
    :cite:`Barten1999`, :cite:`Barten2003`, :cite:`Cowan2004`,
    :cite:`InternationalTelecommunicationUnion2015`,

    Examples
    --------
    >>> contrast_sensitivity_function(u=4)  # doctest: +ELLIPSIS
    360.8691122...
    >>> contrast_sensitivity_function('Barten 1999', u=4)  # doctest: +ELLIPSIS
    360.8691122...
    """

    function = CONTRAST_SENSITIVITY_METHODS[method]

    S = function(**filter_kwargs(function, **kwargs))

    return S


__all__ += ['CONTRAST_SENSITIVITY_METHODS', 'contrast_sensitivity_function']
