# -*- coding: utf-8 -*-
"""
ZCAM Colour Appearance Model
============================

Defines the *ZCAM* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_ZCAM`
-   :attr:`colour.VIEWING_CONDITIONS_ZCAM`
-   :class:`colour.CAM_Specification_ZCAM`
-   :func:`colour.XYZ_to_ZCAM`
-   :func:`colour.ZCAM_to_XYZ`

References
----------
-   :cite:`Safdar2018` : Safdar, M., Hardeberg, J. Y., Kim, Y. J., & Luo, M. R.
    (2018). A Colour Appearance Model based on J z a z b z Colour Space. Color
    and Imaging Conference, 2018(1), 96-101.
    doi:10.2352/ISSN.2169-2629.2018.26.96
-   :cite:`Safdar2021` : Safdar, M., Hardeberg, J. Y., & Ronnier Luo, M.
    (2021). ZCAM, a colour appearance model based on a high dynamic range
    uniform colour space. Optics Express, 29(4), 6036. doi:10.1364/OE.413659
-   :cite:`Zhai2018` : Zhai, Q., & Luo, M. R. (2018). Study of chromatic
    adaptation via neutral white matches on different viewing media. Optics
    Express, 26(6), 7724. doi:10.1364/OE.26.007724
"""

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field
from typing import Union

from colour.adaptation import chromatic_adaptation_Zhai2018
from colour.appearance.ciecam02 import (
    VIEWING_CONDITIONS_CIECAM02,
    degree_of_adaptation,
    hue_angle,
)
from colour.algebra import spow
from colour.colorimetry import CCS_ILLUMINANTS
from colour.models import Izazbz_to_XYZ, XYZ_to_Izazbz, xy_to_XYZ
from colour.utilities import (
    CaseInsensitiveMapping,
    MixinDataclassArray,
    as_float,
    as_float_array,
    as_int_array,
    domain_range_scale,
    from_range_1,
    from_range_degrees,
    has_only_nan,
    ones,
    to_domain_1,
    to_domain_degrees,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'InductionFactors_ZCAM',
    'VIEWING_CONDITIONS_ZCAM',
    'CAM_Specification_ZCAM',
    'XYZ_to_ZCAM',
    'ZCAM_to_XYZ',
]


class InductionFactors_ZCAM(
        namedtuple('InductionFactors_ZCAM', ('F_s', 'F', 'c', 'N_c'))):
    """
    *ZCAM* colour appearance model induction factors.

    Parameters
    ----------
    F_s : numeric or array_like
        Surround impact :math:`F_s`.
    F : numeric or array_like
        Maximum degree of adaptation :math:`F`.
    c : numeric or array_like
        Exponential non-linearity :math:`c`.
    N_c : numeric or array_like
        Chromatic induction factor :math:`N_c`.

    Notes
    -----
    -   The *ZCAM* colour appearance model induction factors are inherited from
        the *CIECAM02* colour appearance model.

    References
    ----------
    :cite:`Safdar2021`
    """


VIEWING_CONDITIONS_ZCAM = CaseInsensitiveMapping({
    'Average':
        InductionFactors_ZCAM(0.69, *VIEWING_CONDITIONS_CIECAM02['Average']),
    'Dim':
        InductionFactors_ZCAM(0.59, *VIEWING_CONDITIONS_CIECAM02['Dim']),
    'Dark':
        InductionFactors_ZCAM(0.525, *VIEWING_CONDITIONS_CIECAM02['Dark']),
})
VIEWING_CONDITIONS_ZCAM.__doc__ = """
Reference *ZCAM* colour appearance model viewing conditions.

References
----------
:cite:`Safdar2021`

VIEWING_CONDITIONS_ZCAM : CaseInsensitiveMapping
    **{'Average', 'Dim', 'Dark'}**
"""

HUE_DATA_FOR_HUE_QUADRATURE = {
    'h_i': np.array([33.44, 89.29, 146.30, 238.36, 393.44]),
    'e_i': np.array([0.68, 0.64, 1.52, 0.77, 0.68]),
    'H_i': np.array([0.0, 100.0, 200.0, 300.0, 400.0])
}


@dataclass
class CAM_ReferenceSpecification_ZCAM(MixinDataclassArray):
    """
    Defines the *ZCAM* colour appearance model reference specification.

    This specification has field names consistent with :cite:`Safdar2021`
    reference.

    Parameters
    ----------
    J_z : numeric or array_like
        Correlate of *Lightness* :math:`J_z`.
    C_z : numeric or array_like
        Correlate of *chroma* :math:`C_z`.
    h_z : numeric or array_like
        *Hue* angle :math:`h_z` in degrees.
    S_z : numeric or array_like
        Correlate of *saturation* :math:`S_z`.
    Q_z : numeric or array_like
        Correlate of *brightness* :math:`Q_z`.
    M_z : numeric or array_like
        Correlate of *colourfulness* :math:`M_z`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    H_z : numeric or array_like
        *Hue* :math:`h` composition :math:`H_z`.
    V_z : numeric or array_like
        Correlate of *vividness* :math:`V_z`.
    K_z : numeric or array_like
        Correlate of *blackness* :math:`K_z`.
    W_z : numeric or array_like
        Correlate of *whiteness* :math:`W_z`.

    References
    ----------
    :cite:`Safdar2021`
    """

    J_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    C_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    h_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    S_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    Q_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    M_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    V_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    K_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    W_z: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


@dataclass
class CAM_Specification_ZCAM(MixinDataclassArray):
    """
    Defines the *ZCAM* colour appearance model specification.

    Parameters
    ----------
    J : numeric or array_like
        *Lightness* :math:`J` is the "brightness of an area (:math:`Q`) judged
        relative to the brightness of a similarly illuminated area that appears
        to be white or highly transmitting (:math:`Q_w`)", i.e.,
        :math:`J = (Q/Q_w)`. It is a visual scale with two well defined levels
        i.e., zero and 100 for a pure black and a reference white,
        respectively. Note that in HDR visual field, samples could have a
        higher luminance than that of the reference white, so the lightness
        could be over 100. Subscripts :math:`s` and :math:`w` are used to
        annotate the sample and the reference white, respectively.
    C : numeric or array_like
        *Chroma* :math:`C` is "colourfulness of an area (:math:`M`) judged as
        a proportion of the brightness of a similarly illuminated area that
        appears white or highly transmitting (:math:`Q_w`)", i.e.,
        :math:`C = (M/Q_w)`. It is an open-end scale with origin as a colour
         in the neutral axis. It can be estimated as the magnitude of the
         chromatic difference between the test colour and a neutral colour
         having the lightness same as the test colour.
    h : numeric or array_like
        *Hue* angle :math:`h` is a scale ranged from :math:`0^{\\circ}` to
        :math:`360^{\\circ} with the hues following rainbow sequence. The same
        distance between pairs of hues in a constant lightness and chroma shows
        the same perceived colour difference.
    s : numeric or array_like
        *Saturation* :math:`s` is the "colourfulness (:math:`M`) of an area
        judged in proportion to its brightness (:math:`Q`)", i.e.,
        :math:`s = (M/Q)`. It can also be defined as the chroma of an area
        judged in proportion to its lightness, i.e., :math:`s = (C/J)`. It is
        an open-end scale with all neutral colours to have saturation of zero.
        For example, the red bricks in a building would exhibit different
        colours when illuminated by daylight. Those (directly) under daylight
        will appear to be bright and colourful, and those under shadow will
        appear darker and less colourful. However, the two areas have the same
        saturation.
    Q : numeric or array_like
        *Brightness* :math:`Q` is an "attribute of a visual perception
        according to which an area appears to emit, or reflect, more or less
        light". It is an open-end scale with origin as pure black or complete
        darkness. It is an absolute scale according to the illumination
        condition i.e., an increase of brightness of an object when the
        illuminance of light is increased. This is a visual phenomenon known as
        Stevens effect.
    M : numeric or array_like
        *Colourfulness* :math:`M` is an "attribute of a visual perception
        according to which the perceived colour of an area appears to be more
        or less chromatic". It is an open-end scale with origin as a neutral
        colour i.e., appearance of no hue. It is an absolute scale according to
        the illumination condition i.e., an increase of colourfulness of an
        object when the illuminance of light is increased. This is a visual
        phenomenon known as Hunt effect.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H_C` is an "attribute of a visual
        perception according to which an area appears to be similar to one of
        the colours: red, yellow, green, and blue, or to a combination of
        adjacent pairs of these colours considered in a closed ring". It has
        a 0-400 scale, i.e., hue quadrature of 0, 100, 200, 300, and 400
        range from unitary red to, yellow, green, blue, and back to red,
        respectively. For example, a cyan colour consists of 50% green and
        50% blue, corresponding to a hue quadrature of 250.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C` used to define the hue
        appearance of a sample. Note that hue circles formed by the equal hue
        angle and equal hue composition appear to be quite different.
    V : numeric or array_like
        *Vividness* :math:`V` is an "attribute of colour used to indicate the
        degree of departure of the  colour (of stimulus) from a neutral black
        colour", i.e., :math:`V = \\sqrt{J^2 + C^2}`. It is an open-end scale
        with origin at pure black. This reflects the visual phenomena of an
        object illuminated by a light to increase both the lightness and the
        chroma.
    K : numeric or array_like
        *Blackness* :math:`K` is a visual attribute according to which an area
        appears to contain more or less black content. It is a scale in the
        Natural Colour System (NCS) and can also be defined in resemblance to a
        pure black. It is an open-end scale with 100 as pure black (luminance
        of 0 :math:`cd/m^2`), i.e.,
        :math:`K = (100 - \\sqrt{J^2 + C^2} = (100 - V)`. The visual effect can
        be illustrated by mixing a black to a colour pigment. The more black
        pigment is added, the higher blackness will be. A blacker colour will
        have less lightness and/or chroma than a less black colour.
    W : numeric or array_like
        *Whiteness* :math:`W` is a visual attribute according to which an area
        appears to contain more or less white content. It is a scale of the NCS
        and can also be defined in resemblance to a pure white. It is an
        open-end scale with 100 as reference white, i.e.,
        :math:`W = (100 - \\sqrt{(100 - J)^2 + C^2} = (100 - D)`. The visual
        effect can be illustrated by mixing a white to a colour pigment. The
        more white pigment is added, the higher whiteness will be. A whiter
        colour will have a lower chroma and higher lightness than the less
        white colour.

    References
    ----------
    :cite:`Safdar2021`
    """

    J: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    C: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    h: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    s: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    Q: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    M: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    H: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    HC: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    V: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    K: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)
    W: Union[float, list, tuple, np.ndarray] = field(
        default_factory=lambda: None)


TVS_D65 = xy_to_XYZ(
    CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])


def XYZ_to_ZCAM(XYZ,
                XYZ_w,
                L_A,
                Y_b,
                surround=VIEWING_CONDITIONS_ZCAM['Average'],
                discount_illuminant=False):
    """
    Computes the *ZCAM* colour appearance model correlates from given *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    XYZ : array_like
        Absolute *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w : array_like
        Absolute *CIE XYZ* tristimulus values of the white under reference
        illuminant.
    L_A : numeric or array_like
        Test adapting field *luminance* :math:`L_A` in :math:`cd/m^2` such as
        :math:`L_A = L_w * Y_b / 100` (where :math:`L_w` is luminance of the
        reference white and :math:`Y_b` is the background luminance factor).
    Y_b : numeric or array_like
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 * L_b / L_w` where :math:`L_w` is the luminance of the
        light source and :math:`L_b` is the luminance of the background. For
        viewing images, :math:`Y_b` can be the average :math:`Y` value for the
        pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximate an :math:`L^*` of 50 is used.
    surround : InductionFactors_ZCAM, optional
        Surround viewing conditions induction factors.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    CAM_Specification_ZCAM
       *ZCAM* colour appearance model specification.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----
    -   *Safdar, Hardeberg and Luo (2021)* does not specify how the chromatic
        adaptation to *CIE Standard Illuminant D65* in *Step 0* should be
        performed. A one-step *Von Kries* chromatic adaptation transform is not
        symmetrical or transitive when a degree of adaptation is involved.
        *Safdar, Hardeberg and Luo (2018)* uses *Zhai and Luo (2018)* two-steps
        chromatic adaptation transform, thus it seems sensible to adopt this
        transform for the *ZCAM* colour appearance model until more information
        is available. It is worth noting that a one-step *Von Kries* chromatic
        adaptation transform with support for degree of adaptation produces
        values closer to the supplemental document compared to the
        *Zhai and Luo (2018)* two-steps chromatic adaptation transform but then
        the *ZCAM* colour appearance model does not round-trip properly.
    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [UN]                  | [UN]          |
    +------------+-----------------------+---------------+
    | ``XYZ_tw`` | [UN]                  | [UN]          |
    +------------+-----------------------+---------------+
    | ``XYZ_rw`` | [UN]                  | [UN]          |
    +------------+-----------------------+---------------+

    +-------------------------------+-----------------------+---------------+
    | **Range**                     | **Scale - Reference** | **Scale - 1** |
    +===============================+=======================+===============+
    | ``CAM_Specification_ZCAM.J``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.C``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.h``  | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.s``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.Q``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.M``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.H``  | [0, 400]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.HC`` | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.V``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.K``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.H``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Safdar2018`, :cite:`Safdar2021`, :cite:`Zhai2018`

    Examples
    --------
    >>> XYZ = np.array([185, 206, 163])
    >>> XYZ_w = np.array([256, 264, 202])
    >>> L_A = 264
    >>> Y_b = 100
    >>> surround = VIEWING_CONDITIONS_ZCAM['Average']
    >>> XYZ_to_ZCAM(XYZ, XYZ_w, L_A, Y_b, surround)
    ... # doctest: +ELLIPSIS
    CAM_Specification_ZCAM(J=92.2504437..., C=3.0216926..., h=196.3245737..., \
s=19.1319556..., Q=321.3408463..., M=10.5256217..., H=237.6114442..., \
HC=None, V=34.7006776..., K=25.8835968..., W=91.6821728...)
    """

    XYZ = to_domain_1(XYZ)
    XYZ_w = to_domain_1(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    F_s, F, c, N_c = surround

    # Step 0 (Forward) - Chromatic adaptation from reference illuminant to
    # "CIE Standard Illuminant D65" illuminant using "CAT02".
    # Computing degree of adaptation :math:`D`.
    D = (degree_of_adaptation(surround.F, L_A)
         if not discount_illuminant else ones(L_A.shape))

    XYZ_D65 = chromatic_adaptation_Zhai2018(
        XYZ, XYZ_w, TVS_D65, D, D, chromatic_adaptation_transform='CAT02')

    # Step 1 (Forward) - Computing factors related with viewing conditions and
    # independent of the test stimulus.
    # Background factor :math:`F_b`
    F_b = np.sqrt(Y_b / Y_w)
    # Luminance level adaptation factor :math:`F_L`
    F_L = 0.171 * spow(L_A, 1 / 3) * (1 - np.exp(-48 / 9 * L_A))

    # Step 2 (Forward) - Computing achromatic response (:math:`I_z` and
    # :math:`I_{z,w}`), redness-greenness (:math:`a_z` and :math:`a_{z,w}`),
    # and yellowness-blueness (:math:`b_z`, :math:`b_{z,w}`).
    with domain_range_scale('ignore'):
        I_z, a_z, b_z = tsplit(XYZ_to_Izazbz(XYZ_D65, method='Safdar 2021'))
        I_z_w, a_z_w, b_z_w = tsplit(
            XYZ_to_Izazbz(XYZ_w, method='Safdar 2021'))

    # Step 3 (Forward) - Computing hue angle :math:`h_z`
    h_z = hue_angle(a_z, b_z)

    # Step 4 (Forward) - Computing hue quadrature :math:`H`.
    H = hue_quadrature(h_z)

    # Computing eccentricity factor :math:`e_z`.
    e_z = 1.015 + np.cos(np.radians(89.038 + h_z % 360))

    # Step 5 (Forward) - Computing brightness :math:`Q_z`,
    # lightness :math:`J_z`, colourfulness :math`M_z`, and chroma :math:`C_z`
    Q_z_p = (1.6 * F_s) / F_b ** 0.12
    Q_z_m = F_s ** 2.2 * F_b ** 0.5 * spow(F_L, 0.2)
    Q_z = 2700 * spow(I_z, Q_z_p) * Q_z_m
    Q_z_w = 2700 * spow(I_z_w, Q_z_p) * Q_z_m

    J_z = 100 * (Q_z / Q_z_w)

    M_z = 100 * (a_z ** 2 + b_z ** 2) ** 0.37 * (
        (spow(e_z, 0.068) * spow(F_L, 0.2)) / (F_b ** 0.1 * spow(I_z_w, 0.78)))

    C_z = 100 * (M_z / Q_z_w)

    # Step 6 (Forward) - Computing saturation :math:`S_z`,
    # vividness :math:`V_z`, blackness :math:`K_z`, and whiteness :math:`W_z`.
    S_z = 100 * spow(F_L, 0.6) * np.sqrt(M_z / Q_z)

    V_z = np.sqrt((J_z - 58) ** 2 + 3.4 * C_z ** 2)

    K_z = 100 - 0.8 * np.sqrt(J_z ** 2 + 8 * C_z ** 2)

    W_z = 100 - np.sqrt((100 - J_z) ** 2 + C_z ** 2)

    return CAM_Specification_ZCAM(
        as_float(from_range_1(J_z)),
        as_float(from_range_1(C_z)),
        as_float(from_range_degrees(h_z)),
        as_float(from_range_1(S_z)),
        as_float(from_range_1(Q_z)),
        as_float(from_range_1(M_z)),
        as_float(from_range_degrees(H, 400)),
        None,
        as_float(from_range_1(V_z)),
        as_float(from_range_1(K_z)),
        as_float(from_range_1(W_z)),
    )


def ZCAM_to_XYZ(specification,
                XYZ_w,
                L_A,
                Y_b,
                surround=VIEWING_CONDITIONS_ZCAM['Average'],
                discount_illuminant=False):
    """
    Converts from *ZCAM* specification to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    specification : CAM_Specification_ZCAM
         *ZCAM* colour appearance model specification.
         Correlate of *Lightness* :math:`J`, correlate of *chroma* :math:`C` or
         correlate of *colourfulness* :math:`M` and *hue* angle :math:`h` in
         degrees must be specified, e.g. :math:`JCh` or :math:`JMh`.
    XYZ_w : array_like
        Absolute *CIE XYZ* tristimulus values of the white under reference
        illuminant.
    L_A : numeric or array_like
        Test adapting field *luminance* :math:`L_A` in :math:`cd/m^2` such as
        :math:`L_A = L_w * Y_b / 100` (where :math:`L_w` is luminance of the
        reference white and :math:`Y_b` is the background luminance factor).
    Y_b : numeric or array_like
        Luminous factor of background :math:`Y_b` such as
        :math:`Y_b = 100 x L_b / L_w` where :math:`L_w` is the luminance of the
        light source and :math:`L_b` is the luminance of the background. For
        viewing images, :math:`Y_b` can be the average :math:`Y` value for the
        pixels in the entire image, or frequently, a :math:`Y` value of 20,
        approximate an :math:`L^*` of 50 is used.
    surround : InductionFactors_ZCAM, optional
        Surround viewing conditions induction factors.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    XYZ : ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If neither *C* or *M* correlates have been defined in the
        ``CAM_Specification_ZCAM`` argument.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----
    -   *Safdar, Hardeberg and Luo (2021)* does not specify how the chromatic
        adaptation to *CIE Standard Illuminant D65* in *Step 0* should be
        performed. A one-step *Von Kries* chromatic adaptation transform is not
        symetrical or transitive when a degree of adptation is involved.
        *Safdar, Hardeberg and Luo (2018)* uses *Zhai and Luo (2018)* two-steps
        chromatic adaptation transform, thus it seems sensible to adopt this
        transform for the *ZCAM* colour appearance model until more information
        is available. It is worth noting that a one-step *Von Kries* chromatic
        adaptation transform with support for degree of adaptation produces
        values closer to the supplemental document compared to the
        *Zhai and Luo (2018)* two-steps chromatic adaptation transform but then
        the *ZCAM* colour appearance model does not round-trip properly.
    -   *Step 4* of the inverse model uses a rounded exponent of 1.3514
        preventing the model to round-trip properly. Given that this
        implementation takes some liberties with respect to the chromatic
        adaptation transform to use, it was deemed appropriate to use an
        exponent value that enables the *ZCAM* colour appearance model to
        round-trip.
    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +-------------------------------+-----------------------+---------------+
    | **Domain**                    | **Scale - Reference** | **Scale - 1** |
    +===============================+=======================+===============+
    | ``CAM_Specification_ZCAM.J``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.C``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.h``  | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.s``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.Q``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.M``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.H``  | [0, 400]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.HC`` | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.V``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.K``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_ZCAM.H``  | [UN]                  | [0, 1]        |
    +-------------------------------+-----------------------+---------------+

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | [UN]                  | [UN]          |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Safdar2018`, :cite:`Safdar2021`, :cite:`Zhai2018`

    Examples
    --------
    >>> specification = CAM_Specification_ZCAM(J=92.250443780723629,
    ...                                        C=3.0216926733329013,
    ...                                        h=196.32457375575581)
    >>> XYZ_w = np.array([256, 264, 202])
    >>> L_A = 264
    >>> Y_b = 100
    >>> surround = VIEWING_CONDITIONS_ZCAM['Average']
    >>> ZCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)
    ... # doctest: +ELLIPSIS
    array([ 185.,  206.,  163.])
    """

    J_z, C_z, h_z, _S_z, _Q_z, M_z, _H, _H_Z, _V_z, _K_z, _W_z = astuple(
        specification)

    J_z = to_domain_1(J_z)
    C_z = to_domain_1(C_z)
    h_z = to_domain_degrees(h_z)
    M_z = to_domain_1(M_z)

    XYZ_w = to_domain_1(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    F_s, F, c, N_c = surround

    # Step 0 (Forward) - Chromatic adaptation from reference illuminant to
    # "CIE Standard Illuminant D65" illuminant using "CAT02".
    # Computing degree of adaptation :math:`D`.
    D = (degree_of_adaptation(surround.F, L_A)
         if not discount_illuminant else ones(L_A.shape))

    # Step 1 (Forward) - Computing factors related with viewing conditions and
    # independent of the test stimulus.
    # Background factor :math:`F_b`
    F_b = np.sqrt(Y_b / Y_w)
    # Luminance level adaptation factor :math:`F_L`
    F_L = 0.171 * spow(L_A, 1 / 3) * (1 - np.exp(-48 / 9 * L_A))

    # Step 2 (Forward) - Computing achromatic response (:math:`I_{z,w}`),
    # redness-greenness (:math:`a_{z,w}`), and yellowness-blueness
    # (:math:`b_{z,w}`).
    with domain_range_scale('ignore'):
        I_z_w, A_z_w, B_z_w = tsplit(
            XYZ_to_Izazbz(XYZ_w, method='Safdar 2021'))

    # Step 1 (Inverse) - Computing achromatic response (:math:`I_z`).
    Q_z_p = (1.6 * F_s) / F_b ** 0.12
    Q_z_m = F_s ** 2.2 * F_b ** 0.5 * spow(F_L, 0.2)
    Q_z_w = 2700 * spow(I_z_w, Q_z_p) * Q_z_m

    I_z_p = (F_b ** 0.12) / (1.6 * F_s)
    I_z_d = 2700 * 100 * Q_z_m

    I_z = spow((J_z * Q_z_w) / I_z_d, I_z_p)

    # Step 2 (Inverse) - Computing chroma :math:`C_z`.
    if has_only_nan(M_z) and not has_only_nan(C_z):
        M_z = (C_z * Q_z_w) / 100
    elif has_only_nan(M_z):
        raise ValueError('Either "C" or "M" correlate must be defined in '
                         'the "CAM_Specification_ZCAM" argument!')

    # Step 3 (Inverse) - Computing hue angle :math:`h_z`
    # :math:`h_z` is currently required as an input.

    # Computing eccentricity factor :math:`e_z`.
    e_z = 1.015 + np.cos(np.radians(89.038 + h_z % 360))
    h_z_r = np.radians(h_z)

    # Step 4 (Inverse) - Computing redness-greenness (:math:`a_z`), and
    # yellowness-blueness (:math:`b_z`).
    # C_z_p_e = 1.3514
    C_z_p_e = 50 / 37
    C_z_p = spow((M_z * spow(I_z_w, 0.78) * F_b ** 0.1) /
                 (100 * e_z ** 0.068 * spow(F_L, 0.2)), C_z_p_e)
    a_z = C_z_p * np.cos(h_z_r)
    b_z = C_z_p * np.sin(h_z_r)

    # Step 5 (Inverse) - Computing tristimulus values :math:`XYZ_{D65}`.
    with domain_range_scale('ignore'):
        XYZ_D65 = Izazbz_to_XYZ(tstack([I_z, a_z, b_z]), method='Safdar 2021')

    XYZ = chromatic_adaptation_Zhai2018(
        XYZ_D65, TVS_D65, XYZ_w, D, D, chromatic_adaptation_transform='CAT02')

    return from_range_1(XYZ)


def hue_quadrature(h):
    """
    Returns the hue quadrature from given hue :math:`h` angle in degrees.

    Parameters
    ----------
    h : numeric or array_like
        Hue :math:`h` angle in degrees.

    Returns
    -------
    numeric or ndarray
        Hue quadrature.

    Examples
    --------
    >>> hue_quadrature(196.3185839)  # doctest: +ELLIPSIS
    237.6052911...
    """

    h = as_float_array(h)

    h_i = HUE_DATA_FOR_HUE_QUADRATURE['h_i']
    e_i = HUE_DATA_FOR_HUE_QUADRATURE['e_i']
    H_i = HUE_DATA_FOR_HUE_QUADRATURE['H_i']

    # :math:`h_p` = :math:`h_z` + 360 if :math:`h_z` < :math:`h_1, i.e. h_i[0]
    h[h <= h_i[0]] += 360
    # *np.searchsorted* returns an erroneous index if a *nan* is used as input.
    h[np.asarray(np.isnan(h))] = 0
    i = as_int_array(np.searchsorted(h_i, h, side='left') - 1)

    h_ii = h_i[i]
    e_ii = e_i[i]
    H_ii = H_i[i]
    h_ii1 = h_i[i + 1]
    e_ii1 = e_i[i + 1]

    H = H_ii + ((100 * (h - h_ii) / e_ii) / (
        (h - h_ii) / e_ii + (h_ii1 - h) / e_ii1))

    return as_float(H)
