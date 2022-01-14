# -*- coding: utf-8 -*-
"""
:math:`IC_TC_P` Colour Encoding
===============================

Defines the :math:`IC_TC_P` colour encoding related transformations:

-   :func:`colour.RGB_to_ICtCp`
-   :func:`colour.ICtCp_to_RGB`
-   :func:`colour.XYZ_to_ICtCp`
-   :func:`colour.ICtCp_to_XYZ`

References
----------
-   :cite:`Dolby2016a` : Dolby. (2016). WHAT IS ICtCp? - INTRODUCTION.
    https://www.dolby.com/us/en/technologies/dolby-vision/ICtCp-white-paper.pdf
-   :cite:`InternationalTelecommunicationUnion2018` : International
    Telecommunication Union. (2018). Recommendation ITU-R BT.2100-2 - Image
    parameter values for high dynamic range television for use in production
    and international programme exchange.
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2100-2-201807-I!!PDF-E.pdf
-   :cite:`Lu2016c` : Lu, T., Pu, F., Yin, P., Chen, T., Husak, W., Pytlarz,
    J., Atkins, R., Froehlich, J., & Su, G.-M. (2016). ITP Colour Space and Its
    Compression Performance for High Dynamic Range and Wide Colour Gamut Video
    Distribution. ZTE Communications, 14(1), 32-38.
"""

from __future__ import annotations

import numpy as np

from colour.algebra import vector_dot
from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import ArrayLike, Floating, Literal, NDArray, Union
from colour.models.rgb import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_RGB
from colour.models.rgb.transfer_functions import (
    eotf_ST2084,
    eotf_inverse_ST2084,
    oetf_HLG_BT2100,
    oetf_inverse_HLG_BT2100,
)
from colour.utilities import (
    domain_range_scale,
    from_range_1,
    to_domain_1,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_ICTCP_RGB_TO_LMS',
    'MATRIX_ICTCP_LMS_TO_RGB',
    'MATRIX_ICTCP_LMS_P_TO_ICTCP',
    'MATRIX_ICTCP_ICTCP_TO_LMS_P',
    'MATRIX_ICTCP_LMS_P_TO_ICTCP_HLG_BT2100_2',
    'MATRIX_ICTCP_ICTCP_TO_LMS_P_HLG_BT2100_2',
    'RGB_to_ICtCp',
    'ICtCp_to_RGB',
    'XYZ_to_ICtCp',
    'ICtCp_to_XYZ',
]

MATRIX_ICTCP_RGB_TO_LMS: NDArray = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
]) / 4096
"""
*ITU-R BT.2020* colourspace to normalised cone responses matrix.
"""

MATRIX_ICTCP_LMS_TO_RGB: NDArray = np.linalg.inv(MATRIX_ICTCP_RGB_TO_LMS)
"""
:math:`IC_TC_P` colourspace normalised cone responses to *ITU-R BT.2020*
colourspace matrix.
"""

MATRIX_ICTCP_LMS_P_TO_ICTCP: NDArray = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
]) / 4096
"""
:math:`LMS_p` *SMPTE ST 2084:2014* encoded normalised cone responses to
:math:`IC_TC_P` colour encoding matrix.
"""

MATRIX_ICTCP_ICTCP_TO_LMS_P: NDArray = np.linalg.inv(
    MATRIX_ICTCP_LMS_P_TO_ICTCP)
"""
:math:`IC_TC_P` colour encoding to :math:`LMS_p` *SMPTE ST 2084:2014* encoded
normalised cone responses matrix.
"""

MATRIX_ICTCP_LMS_P_TO_ICTCP_HLG_BT2100_2: NDArray = np.array([
    [2048, 2048, 0],
    [3625, -7465, 3840],
    [9500, -9212, -288],
]) / 4096
"""
:math:`LMS_p` *SMPTE ST 2084:2014* encoded normalised cone responses to
:math:`IC_TC_P` colour encoding matrix as given in *ITU-R BT.2100-2*.
"""

MATRIX_ICTCP_ICTCP_TO_LMS_P_HLG_BT2100_2: NDArray = np.linalg.inv(
    MATRIX_ICTCP_LMS_P_TO_ICTCP_HLG_BT2100_2)
"""
:math:`IC_TC_P` colour encoding to :math:`LMS_p` *SMPTE ST 2084:2014* encoded
normalised cone responses matrix as given in *ITU-R BT.2100-2*.
"""


def RGB_to_ICtCp(
        RGB: ArrayLike,
        method: Union[Literal['Dolby 2016', 'ITU-R BT.2100-1 HLG',
                              'ITU-R BT.2100-1 PQ', 'ITU-R BT.2100-2 HLG',
                              'ITU-R BT.2100-2 PQ'], str] = 'Dolby 2016',
        L_p: Floating = 10000) -> NDArray:
    """
    Converts from *ITU-R BT.2020* colourspace to :math:`IC_TC_P` colour
    encoding.

    Parameters
    ----------
    RGB
        *ITU-R BT.2020* colourspace array.
    method
        Computation method. *Recommendation ITU-R BT.2100* defines multiple
        variants of the :math:`IC_TC_P` colour encoding:

        -   *ITU-R BT.2100-1*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and the :math:`IC_TC_P` matrix
                from :cite:`Dolby2016a`: *ITU-R BT.2100-1 HLG* method.

        -   *ITU-R BT.2100-2*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and a custom :math:`IC_TC_P`
                matrix from :cite:`InternationalTelecommunicationUnion2018`:
                *ITU-R BT.2100-2 HLG* method.

    L_p
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`IC_TC_P` colour encoding array.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----
    -   The *ITU-R BT.2100-1 PQ* and *ITU-R BT.2100-2 PQ* methods are aliases
        for the *Dolby 2016* method.
    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations. The effective domain of *SMPTE ST 2084:2014*
        inverse electro-optical transfer function (EOTF) is
        [0.0001, 10000].

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``RGB``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICtCp``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_ICtCp(RGB)  # doctest: +ELLIPSIS
    array([ 0.0735136...,  0.0047525...,  0.0935159...])
    >>> RGB_to_ICtCp(RGB, method='ITU-R BT.2100-2 HLG')  # doctest: +ELLIPSIS
    array([ 0.6256789..., -0.0198449...,  0.3591125...])
    """

    RGB = to_domain_1(RGB)
    method = validate_method(method, [
        'Dolby 2016', 'ITU-R BT.2100-1 HLG', 'ITU-R BT.2100-1 PQ',
        'ITU-R BT.2100-2 HLG', 'ITU-R BT.2100-2 PQ'
    ])

    is_hlg_method = 'hlg' in method
    is_BT2100_2_method = '2100-2' in method

    LMS = vector_dot(MATRIX_ICTCP_RGB_TO_LMS, RGB)

    with domain_range_scale('ignore'):
        LMS_p = (oetf_HLG_BT2100(LMS)
                 if is_hlg_method else eotf_inverse_ST2084(LMS, L_p))

    ICtCp = (vector_dot(MATRIX_ICTCP_LMS_P_TO_ICTCP_HLG_BT2100_2, LMS_p)
             if (is_hlg_method and is_BT2100_2_method) else vector_dot(
                 MATRIX_ICTCP_LMS_P_TO_ICTCP, LMS_p))

    return from_range_1(ICtCp)


def ICtCp_to_RGB(
        ICtCp: ArrayLike,
        method: Union[Literal['Dolby 2016', 'ITU-R BT.2100-1 HLG',
                              'ITU-R BT.2100-1 PQ', 'ITU-R BT.2100-2 HLG',
                              'ITU-R BT.2100-2 PQ'], str] = 'Dolby 2016',
        L_p: Floating = 10000) -> NDArray:
    """
    Converts from :math:`IC_TC_P` colour encoding to *ITU-R BT.2020*
    colourspace.

    Parameters
    ----------
    ICtCp
        :math:`IC_TC_P` colour encoding array.
    method
        Computation method. *Recommendation ITU-R BT.2100* defines multiple
        variants of the :math:`IC_TC_P` colour encoding:

        -   *ITU-R BT.2100-1*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and the :math:`IC_TC_P` matrix
                from :cite:`Dolby2016a`: *ITU-R BT.2100-1 HLG* method.

        -   *ITU-R BT.2100-2*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and a custom :math:`IC_TC_P`
                matrix from :cite:`InternationalTelecommunicationUnion2018`:
                *ITU-R BT.2100-2 HLG* method.

    L_p
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    :class:`numpy.ndarray`
        *ITU-R BT.2020* colourspace array.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----

    -   The *ITU-R BT.2100-1 PQ* and *ITU-R BT.2100-2 PQ* methods are aliases
        for the *Dolby 2016* method.
    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICtCp``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``RGB``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> ICtCp = np.array([0.07351364, 0.00475253, 0.09351596])
    >>> ICtCp_to_RGB(ICtCp)  # doctest: +ELLIPSIS
    array([ 0.4562052...,  0.0308107...,  0.0409195...])
    >>> ICtCp = np.array([0.62567899, -0.01984490, 0.35911259])
    >>> ICtCp_to_RGB(ICtCp, method='ITU-R BT.2100-2 HLG')  # doctest: +ELLIPSIS
    array([ 0.4562052...,  0.0308107...,  0.0409195...])
    """

    ICtCp = to_domain_1(ICtCp)
    method = validate_method(method, [
        'Dolby 2016', 'ITU-R BT.2100-1 HLG', 'ITU-R BT.2100-1 PQ',
        'ITU-R BT.2100-2 HLG', 'ITU-R BT.2100-2 PQ'
    ])

    is_hlg_method = 'hlg' in method
    is_BT2100_2_method = '2100-2' in method

    LMS_p = (vector_dot(MATRIX_ICTCP_ICTCP_TO_LMS_P_HLG_BT2100_2, ICtCp)
             if (is_hlg_method and is_BT2100_2_method) else vector_dot(
                 MATRIX_ICTCP_ICTCP_TO_LMS_P, ICtCp))

    with domain_range_scale('ignore'):
        LMS = (oetf_inverse_HLG_BT2100(LMS_p)
               if is_hlg_method else eotf_ST2084(LMS_p, L_p))

    RGB = vector_dot(MATRIX_ICTCP_LMS_TO_RGB, LMS)

    return from_range_1(RGB)


def XYZ_to_ICtCp(
        XYZ: ArrayLike,
        illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65'],
        chromatic_adaptation_transform: Union[Literal[
            'Bianco 2010', 'Bianco PC 2010', 'Bradford', 'CAT02 Brill 2008',
            'CAT02', 'CAT16', 'CMCCAT2000', 'CMCCAT97', 'Fairchild', 'Sharp',
            'Von Kries', 'XYZ Scaling'], str] = 'CAT02',
        method: Union[Literal['Dolby 2016', 'ITU-R BT.2100-1 HLG',
                              'ITU-R BT.2100-1 PQ', 'ITU-R BT.2100-2 HLG',
                              'ITU-R BT.2100-2 PQ'], str] = 'Dolby 2016',
        L_p: Floating = 10000) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to :math:`IC_TC_P` colour
    encoding.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.
    method
        Computation method. *Recommendation ITU-R BT.2100* defines multiple
        variants of the :math:`IC_TC_P` colour encoding:

        -   *ITU-R BT.2100-1*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and the :math:`IC_TC_P` matrix
                from :cite:`Dolby2016a`: *ITU-R BT.2100-1 HLG* method.

        -   *ITU-R BT.2100-2*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and a custom :math:`IC_TC_P`
                matrix from :cite:`InternationalTelecommunicationUnion2018`:
                *ITU-R BT.2100-2 HLG* method.

    L_p
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`IC_TC_P` colour encoding array.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----

    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
    -   The *ITU-R BT.2100-1 PQ* and *ITU-R BT.2100-2 PQ* methods are aliases
        for the *Dolby 2016* method.
        and *1* scales are only indicative that the data is not affected by
        scale transformations. The effective domain of *SMPTE ST 2084:2014*
        inverse electro-optical transfer function (EOTF) is
        [0.0001, 10000].

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``XYZ``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICtCp``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_ICtCp(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0685809..., -0.0028384...,  0.0602098...])
    >>> XYZ_to_ICtCp(XYZ, method='ITU-R BT.2100-2 HLG')  # doctest: +ELLIPSIS
    array([ 0.5924279..., -0.0374073...,  0.2512267...])
    """

    BT2020 = RGB_COLOURSPACES['ITU-R BT.2020']

    RGB = XYZ_to_RGB(
        XYZ,
        illuminant,
        BT2020.whitepoint,
        BT2020.matrix_XYZ_to_RGB,
        chromatic_adaptation_transform,
    )

    return RGB_to_ICtCp(RGB, method, L_p)


def ICtCp_to_XYZ(
        ICtCp: ArrayLike,
        illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65'],
        chromatic_adaptation_transform: Union[Literal[
            'Bianco 2010', 'Bianco PC 2010', 'Bradford', 'CAT02 Brill 2008',
            'CAT02', 'CAT16', 'CMCCAT2000', 'CMCCAT97', 'Fairchild', 'Sharp',
            'Von Kries', 'XYZ Scaling'], str] = 'CAT02',
        method: Union[Literal['Dolby 2016', 'ITU-R BT.2100-1 HLG',
                              'ITU-R BT.2100-1 PQ', 'ITU-R BT.2100-2 HLG',
                              'ITU-R BT.2100-2 PQ'], str] = 'Dolby 2016',
        L_p: Floating = 10000) -> NDArray:
    """
    Converts from :math:`IC_TC_P` colour encoding to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    ICtCp
        :math:`IC_TC_P` colour encoding array.
    illuminant
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.
    method
        Computation method. *Recommendation ITU-R BT.2100* defines multiple
        variants of the :math:`IC_TC_P` colour encoding:

        -   *ITU-R BT.2100-1*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and the :math:`IC_TC_P` matrix
                from :cite:`Dolby2016a`: *ITU-R BT.2100-1 HLG* method.

        -   *ITU-R BT.2100-2*

            -   *SMPTE ST 2084:2014* inverse electro-optical transfer
                function (EOTF) and the :math:`IC_TC_P` matrix from
                :cite:`Dolby2016a`: *Dolby 2016*, *ITU-R BT.2100-1 PQ*,
                *ITU-R BT.2100-2 PQ* methods.
            -   *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
                transfer function (OETF) and a custom :math:`IC_TC_P`
                matrix from :cite:`InternationalTelecommunicationUnion2018`:
                *ITU-R BT.2100-2 HLG* method.

    L_p
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----

    -   The *ITU-R BT.2100-1 PQ* and *ITU-R BT.2100-2 PQ* methods are aliases
        for the *Dolby 2016* method.
    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICtCp``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``XYZ``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> ICtCp = np.array([0.06858097, -0.00283842, 0.06020983])
    >>> ICtCp_to_XYZ(ICtCp)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    >>> ICtCp = np.array([0.59242792, -0.03740730, 0.25122675])
    >>> ICtCp_to_XYZ(ICtCp, method='ITU-R BT.2100-2 HLG')  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    RGB = ICtCp_to_RGB(ICtCp, method, L_p)

    BT2020 = RGB_COLOURSPACES['ITU-R BT.2020']

    XYZ = RGB_to_XYZ(
        RGB,
        BT2020.whitepoint,
        illuminant,
        BT2020.matrix_RGB_to_XYZ,
        chromatic_adaptation_transform,
    )

    return XYZ
