# -*- coding: utf-8 -*-
"""
Colour Quality Scale
====================

Defines *Colour Quality Scale* (CQS) computation objects:

-   :class:`colour.quality.CQS_Specification`
-   :func:`colour.colour_quality_scale`

See Also
--------
`Colour Quality Scale Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/quality/cqs.ipynb>`_

References
----------
-   :cite:`Davis2010a` : Davis, W., & Ohno, Y. (2010). Color quality scale.
    Optical Engineering, 49(3), 33602. doi:10.1117/1.3360335
-   :cite:`Ohno2008a` : Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4.
    Retrieved from https://drive.google.com/file/d/\
1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?usp=sharing
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.algebra import euclidean_distance
from colour.colorimetry import (
    ASTME30815_PRACTISE_SHAPE, sd_CIE_illuminant_D_series, ILLUMINANTS,
    STANDARD_OBSERVERS_CMFS, sd_blackbody, sd_to_XYZ)
from colour.quality.dataset.vs import VS_INDEXES_TO_NAMES, VS_SDS
from colour.models import (Lab_to_LCHab, UCS_to_uv, XYZ_to_Lab, XYZ_to_UCS,
                           XYZ_to_xy, xy_to_XYZ)
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Ohno2013
from colour.adaptation import chromatic_adaptation_VonKries
from colour.utilities import as_float_array, domain_range_scale, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'D65_GAMUT_AREA', 'VS_ColorimetryData', 'VS_ColourQualityScaleData',
    'CQS_Specification', 'colour_quality_scale', 'gamut_area',
    'vs_colorimetry_data', 'CCT_factor', 'scale_conversion', 'delta_E_RMS',
    'colour_quality_scales'
]

D65_GAMUT_AREA = 8210


class VS_ColorimetryData(
        namedtuple('VS_ColorimetryData', ('name', 'XYZ', 'Lab', 'C'))):
    """
    Defines the the class storing *VS test colour samples* colorimetry data.
    """


class VS_ColourQualityScaleData(
        namedtuple('VS_ColourQualityScaleData',
                   ('name', 'Q_a', 'D_C_ab', 'D_E_ab', 'D_Ep_ab'))):
    """
    Defines the the class storing *VS test colour samples* colour quality
    scale data.
    """


class CQS_Specification(
        namedtuple('CQS_Specification', ('name', 'Q_a', 'Q_f', 'Q_p', 'Q_g',
                                         'Q_d', 'Q_as', 'colorimetry_data'))):
    """
    Defines the *Colour Quality Scale* (CQS) colour quality specification.

    Parameters
    ----------
    name : unicode
        Name of the test spectral distribution.
    Q_a : numeric
        Colour quality scale :math:`Q_a`.
    Q_f : numeric
        Colour fidelity scale :math:`Q_f` intended to evaluate the fidelity
        of object colour appearances (compared to the reference illuminant of
        the same correlated colour temperature and illuminance).
    Q_p : numeric
        Colour preference scale :math:`Q_p` similar to colour quality scale
        :math:`Q_a` but placing additional weight on preference of object
        colour appearance. This metric is based on the notion that increases
        in chroma are generally preferred and should be rewarded.
    Q_g : numeric
         Gamut area scale :math:`Q_g` representing the relative gamut formed
         by the (:math:`a^*`, :math:`b^*`) coordinates of the 15 samples
         illuminated by the test light source in the *CIE L\\*a\\*b\\** object
         colourspace.
    Q_d : numeric
        Relative gamut area scale :math:`Q_d`.
    Q_as : dict
        Individual *Colour Quality Scale* (CQS) data for each sample.
    colorimetry_data : tuple
        Colorimetry data for the test and reference computations.

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`
    """


def colour_quality_scale(sd_test, additional_data=False):
    """
    Returns the *Colour Quality Scale* (CQS) of given spectral
    distribution.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    additional_data : bool, optional
        Whether to output additional data.

    Returns
    -------
    numeric or CQS_Specification
        Color quality scale.

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`

    Examples
    --------
    >>> from colour import ILLUMINANTS_SDS
    >>> sd = ILLUMINANTS_SDS['FL2']
    >>> colour_quality_scale(sd)  # doctest: +ELLIPSIS
    64.6863391...
    """

    cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].copy(
    ).trim(ASTME30815_PRACTISE_SHAPE)

    shape = cmfs.shape
    sd_test = sd_test.copy().align(shape)
    vs_sds = {sd.name: sd.copy().align(shape) for sd in VS_SDS.values()}

    with domain_range_scale('1'):
        XYZ = sd_to_XYZ(sd_test, cmfs)

    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, _D_uv = uv_to_CCT_Ohno2013(uv)

    if CCT < 5000:
        sd_reference = sd_blackbody(CCT, shape)
    else:
        xy = CCT_to_xy_CIE_D(CCT)
        sd_reference = sd_CIE_illuminant_D_series(xy)
        sd_reference.align(shape)

    test_vs_colorimetry_data = vs_colorimetry_data(
        sd_test, sd_reference, vs_sds, cmfs, chromatic_adaptation=True)

    reference_vs_colorimetry_data = vs_colorimetry_data(
        sd_reference, sd_reference, vs_sds, cmfs)

    XYZ_r = sd_to_XYZ(sd_reference, cmfs)
    XYZ_r /= XYZ_r[1]
    CCT_f = CCT_factor(reference_vs_colorimetry_data, XYZ_r)

    Q_as = colour_quality_scales(test_vs_colorimetry_data,
                                 reference_vs_colorimetry_data, CCT_f)

    D_E_RMS = delta_E_RMS(Q_as, 'D_E_ab')
    D_Ep_RMS = delta_E_RMS(Q_as, 'D_Ep_ab')

    Q_a = scale_conversion(D_Ep_RMS, CCT_f)
    Q_f = scale_conversion(D_E_RMS, CCT_f, 2.928)

    p_delta_C = np.average(
        [sample_data.D_C_ab if sample_data.D_C_ab > 0 else 0
         for sample_data in Q_as.values()])  # yapf: disable
    Q_p = 100 - 3.6 * (D_Ep_RMS - p_delta_C)

    G_t = gamut_area(
        [vs_CQS_data.Lab for vs_CQS_data in test_vs_colorimetry_data])
    G_r = gamut_area(
        [vs_CQS_data.Lab for vs_CQS_data in reference_vs_colorimetry_data])

    Q_g = G_t / D65_GAMUT_AREA * 100
    Q_d = G_t / G_r * CCT_f * 100

    if additional_data:
        return CQS_Specification(
            sd_test.name, Q_a, Q_f, Q_p, Q_g, Q_d, Q_as,
            (test_vs_colorimetry_data, reference_vs_colorimetry_data))
    else:
        return Q_a


def gamut_area(Lab):
    """
    Returns the gamut area :math:`G` covered by given *CIE L\\*a\\*b\\**
    matrices.

    Parameters
    ----------
    Lab : array_like
        *CIE L\\*a\\*b\\** colourspace matrices.

    Returns
    -------
    numeric
        Gamut area :math:`G`.

    Examples
    --------
    >>> Lab = [
    ...     np.array([39.94996006, 34.59018231, -19.86046321]),
    ...     np.array([38.88395498, 21.44348519, -34.87805301]),
    ...     np.array([36.60576301, 7.06742454, -43.21461177]),
    ...     np.array([46.60142558, -15.90481586, -34.64616865]),
    ...     np.array([56.50196523, -29.54655550, -20.50177194]),
    ...     np.array([55.73912101, -43.39520959, -5.08956953]),
    ...     np.array([56.20776870, -53.68997662, 20.21134410]),
    ...     np.array([66.16683122, -38.64600327, 42.77396631]),
    ...     np.array([76.72952110, -23.92148210, 61.04740432]),
    ...     np.array([82.85370708, -3.98679065, 75.43320144]),
    ...     np.array([69.26458861, 13.11066359, 68.83858372]),
    ...     np.array([69.63154351, 28.24532497, 59.45609803]),
    ...     np.array([61.26281449, 40.87950839, 44.97606172]),
    ...     np.array([41.62567821, 57.34129516, 27.46718170]),
    ...     np.array([40.52565174, 48.87449192, 3.45121680])
    ... ]
    >>> gamut_area(Lab)  # doctest: +ELLIPSIS
    8335.9482018...
    """

    Lab = as_float_array(Lab)
    Lab_s = np.roll(np.copy(Lab), -3)

    _L, a, b = tsplit(Lab)
    _L_s, a_s, b_s = tsplit(Lab_s)

    A = np.linalg.norm(Lab[..., 1:3], axis=-1)
    B = np.linalg.norm(Lab_s[..., 1:3], axis=-1)
    C = np.linalg.norm(np.dstack([a_s - a, b_s - b]), axis=-1)
    t = (A + B + C) / 2
    S = np.sqrt(t * (t - A) * (t - B) * (t - C))

    return np.sum(S)


def vs_colorimetry_data(sd_test,
                        sd_reference,
                        sds_vs,
                        cmfs,
                        chromatic_adaptation=False):
    """
    Returns the *VS test colour samples* colorimetry data.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    sd_reference : SpectralDistribution
        Reference spectral distribution.
    sds_vs : dict
        *VS test colour samples* spectral distributions.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    chromatic_adaptation : bool, optional
        Perform chromatic adaptation.

    Returns
    -------
    list
        *VS test colour samples* colorimetry data.
    """

    XYZ_t = sd_to_XYZ(sd_test, cmfs)
    XYZ_t /= XYZ_t[1]

    XYZ_r = sd_to_XYZ(sd_reference, cmfs)
    XYZ_r /= XYZ_r[1]
    xy_r = XYZ_to_xy(XYZ_r)

    vs_data = []
    for _key, value in sorted(VS_INDEXES_TO_NAMES.items()):
        sd_vs = sds_vs[value]

        with domain_range_scale('1'):
            XYZ_vs = sd_to_XYZ(sd_vs, cmfs, sd_test)

        if chromatic_adaptation:
            XYZ_vs = chromatic_adaptation_VonKries(
                XYZ_vs, XYZ_t, XYZ_r, transform='CMCCAT2000')

        Lab_vs = XYZ_to_Lab(XYZ_vs, illuminant=xy_r)
        _L_vs, C_vs, _Hab = Lab_to_LCHab(Lab_vs)

        vs_data.append(VS_ColorimetryData(sd_vs.name, XYZ_vs, Lab_vs, C_vs))
    return vs_data


def CCT_factor(reference_data, XYZ_r):
    """
    Returns the correlated colour temperature factor penalizing lamps with
    extremely low correlated colour temperatures.

    Parameters
    ----------
    reference_data : VS_ColorimetryData
        Reference colorimetry data.
    XYZ_r : array_like
        *CIE XYZ* tristimulus values for reference.

    Returns
    -------
    numeric
        Correlated colour temperature factor.
    """

    xy_w = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    XYZ_w = xy_to_XYZ(xy_w)

    Labs = []
    for vs_colorimetry_data_ in reference_data:
        _name, XYZ, _Lab, _C = vs_colorimetry_data_
        XYZ_a = chromatic_adaptation_VonKries(
            XYZ, XYZ_r, XYZ_w, transform='CMCCAT2000')

        Lab = XYZ_to_Lab(XYZ_a, illuminant=xy_w)
        Labs.append(Lab)

    G_r = gamut_area(Labs) / D65_GAMUT_AREA
    CCT_f = 1 if G_r > 1 else G_r

    return CCT_f


def scale_conversion(D_E_ab, CCT_f, scaling_f=3.104):
    """
    Returns the *Colour Quality Scale* (CQS) for given :math:`\\Delta E_{ab}`
    value and given correlated colour temperature penalizing factor.

    Parameters
    ----------
    D_E_ab : numeric
        :math:`\\Delta E_{ab}` value.
    CCT_f : numeric
        Correlated colour temperature penalizing factor.
    scaling_f : numeric, optional
        Scaling factor constant.

    Returns
    -------
    numeric
        *Colour Quality Scale* (CQS).
    """

    Q_a = 10 * np.log(np.exp((100 - scaling_f * D_E_ab) / 10) + 1) * CCT_f

    return Q_a


def delta_E_RMS(cqs_data, attribute):
    """
    Computes the root-mean-square average for given *Colour Quality Scale*
    (CQS) data.

    Parameters
    ----------
    cqs_data : VS_ColourQualityScaleData
        *Colour Quality Scale* (CQS) data.
    attribute : unicode
        Colorimetry data attribute to use to compute the root-mean-square
        average.

    Returns
    -------
    numeric
        Root-mean-square average.
    """

    return np.sqrt(1 / len(cqs_data) * np.sum(
        [getattr(sample_data, attribute) ** 2
         for sample_data in cqs_data.values()]))  # yapf: disable


def colour_quality_scales(test_data, reference_data, CCT_f):
    """
    Returns the *VS test colour samples* rendering scales.

    Parameters
    ----------
    test_data : list
        Test data.
    reference_data : list
        Reference data.
    CCT_f : numeric
        Factor penalizing lamps with extremely low correlated colour
        temperatures.

    Returns
    -------
    dict
        *VS Test colour samples* colour rendering scales.
    """

    Q_as = {}
    for i, _ in enumerate(test_data):
        D_C_ab = test_data[i].C - reference_data[i].C
        D_E_ab = euclidean_distance(test_data[i].Lab, reference_data[i].Lab)

        if D_C_ab > 0:
            D_Ep_ab = np.sqrt(D_E_ab ** 2 - D_C_ab ** 2)
        else:
            D_Ep_ab = D_E_ab

        Q_a = scale_conversion(D_Ep_ab, CCT_f)

        Q_as[i + 1] = VS_ColourQualityScaleData(test_data[i].name, Q_a, D_C_ab,
                                                D_E_ab, D_Ep_ab)
    return Q_as
