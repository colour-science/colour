# -*- coding: utf-8 -*-
"""
Corresponding Chromaticities Prediction
=======================================

Defines objects to compute corresponding chromaticities prediction.

See Also
--------
`Corresponding Chromaticities Prediction Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/corresponding/prediction.ipynb>`_

References
----------
-   :cite:`Breneman1987b` : Breneman, E. J. (1987). Corresponding
    chromaticities for different states of adaptation to complex visual fields.
    Journal of the Optical Society of America A, 4(6), 1115.
    doi:10.1364/JOSAA.4.001115
-   :cite:`CIETC1-321994b` : CIE TC 1-32. (1994). CIE 109-1994 A Method of
    Predicting Corresponding Colours under Different Chromatic and Illuminance
    Adaptations. ISBN:978-3-900734-51-0
-   :cite:`Fairchild1991a` : Fairchild, M. D. (1991). Formulation and testing
    of an incomplete-chromatic-adaptation model. Color Research & Application,
    16(4), 243-250. doi:10.1002/col.5080160406
-   :cite:`Fairchild2013s` : Fairchild, M. D. (2013). FAIRCHILD'S 1990 MODEL.
    In Color Appearance Models (3rd ed., pp. 4418-4495). Wiley. ISBN:B00DAYO8E2
-   :cite:`Fairchild2013t` : Fairchild, M. D. (2013). Chromatic Adaptation
    Models. In Color Appearance Models (3rd ed., pp. 4179-4252). Wiley.
    ISBN:B00DAYO8E2
-   :cite:`Li2002a` : Li, C., Luo, M. R., Rigg, B., & Hunt, R. W. G. (2002).
    CMC 2000 chromatic adaptation transform: CMCCAT2000. Color Research &
    Application, 27(1), 49-58. doi:10.1002/col.10005
-   :cite:`Luo1999` : Luo, M. R., & Rhodes, P. A. (1999). Corresponding-colour
    datasets. Color Research & Application, 24(4), 295-296.
    doi:10.1002/(SICI)1520-6378(199908)24:4<295::AID-COL10>3.0.CO;2-K
-   :cite:`Westland2012k` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT2000. In Computational Colour Science Using MATLAB
    (2nd ed., pp. 83-86). ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.adaptation import (
    chromatic_adaptation_CIE1994, chromatic_adaptation_CMCCAT2000,
    chromatic_adaptation_Fairchild1990, chromatic_adaptation_VonKries)
from colour.corresponding import (
    BRENEMAN_EXPERIMENTS, BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES)
from colour.models import (Luv_to_uv, Luv_uv_to_xy, XYZ_to_Luv, XYZ_to_xy,
                           xy_to_XYZ, xyY_to_XYZ)
from colour.utilities import (CaseInsensitiveMapping, domain_range_scale,
                              filter_kwargs, is_numeric)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CorrespondingColourDataset', 'CorrespondingChromaticitiesPrediction',
    'convert_experiment_results_Breneman1987',
    'corresponding_chromaticities_prediction_Fairchild1990',
    'corresponding_chromaticities_prediction_CIE1994',
    'corresponding_chromaticities_prediction_CMCCAT2000',
    'corresponding_chromaticities_prediction_VonKries',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'corresponding_chromaticities_prediction'
]


class CorrespondingColourDataset(
        namedtuple('CorrespondingColourDataset',
                   ('name', 'XYZ_r', 'XYZ_t', 'XYZ_cr', 'XYZ_ct', 'Y_r', 'Y_t',
                    'B_r', 'B_t', 'metadata'))):
    """
    Defines a corresponding colour dataset.

    Parameters
    ----------
    name : unicode
        Corresponding colour dataset name.
    XYZ_r : array_like
        *CIE XYZ* tristimulus values of the reference illuminant.
    XYZ_t : array_like
        *CIE XYZ* tristimulus values of the test illuminant.
    XYZ_cr : array_like
        Corresponding *CIE XYZ* tristimulus values under the reference
        illuminant.
    XYZ_ct : array_like
        Corresponding *CIE XYZ* tristimulus values under the test illuminant.
    Y_r : numeric
        Reference white luminance :math:`Y_r` in :math:`cd/m^2`.
    Y_t : numeric
        Test white luminance :math:`Y_t` in :math:`cd/m^2`.
    B_r : numeric
         Luminance factor :math:`B_r` of reference achromatic background as
         percentage.
    B_t : numeric
         Luminance factor :math:`B_t` of test achromatic background as
         percentage.
    metadata : dict
        Dataset metadata.

    Notes
    -----
    -   This class is compatible with *Luo and Rhodes (1999)*
        *Corresponding-Colour Datasets* datasets.

    References
    ----------
    :cite:`Luo1999`
    """


class CorrespondingChromaticitiesPrediction(
        namedtuple('CorrespondingChromaticitiesPrediction',
                   ('name', 'uv_t', 'uv_m', 'uv_p'))):
    """
    Defines a chromatic adaptation model prediction.

    Parameters
    ----------
    name : unicode
        Test colour name.
    uv_t : array_like, (2,)
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    uv_m : array_like, (2,)
        Chromaticity coordinates :math:`uv_m^p` of matching colour.
    uv_p : array_like, (2,)
        Chromaticity coordinates :math:`uv_p^p` of predicted colour.
    """


def convert_experiment_results_Breneman1987(experiment):
    """
    Converts *Breneman (1987)* experiment results to a
    :class:`colour.CorrespondingColourDataset` class instance.

    Parameters
    ----------
    experiment : integer
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.

    Returns
    -------
    CorrespondingColourDataset
        :class:`colour.CorrespondingColourDataset` class instance.

    Examples
    --------
    >>> from pprint import pprint
    >>> pprint(tuple(convert_experiment_results_Breneman1987(2)))
    ... # doctest: +ELLIPSIS
    (2,
     array([ 0.9582463...,  1.        ,  0.9436325...]),
     array([ 0.9587332...,  1.        ,  0.4385796...]),
     array([[ 388.125     ,  405.        ,  345.625     ],
           [ 266.8957925...,  135.        ,   28.5983365...],
           [ 474.5717821...,  405.        ,  222.75     ...],
           [ 538.3899082...,  405.        ,   24.8944954...],
           [ 178.7430167...,  135.        ,   19.6089385...],
           [ 436.6749547...,  405.        ,   26.5483725...],
           [ 124.7746282...,  135.        ,   36.1965613...],
           [  77.0794172...,  135.        ,   60.5850563...],
           [ 279.9390889...,  405.        ,  455.8395127...],
           [ 149.5808157...,  135.        ,  498.7046827...],
           [ 372.1113689...,  405.        ,  669.9883990...],
           [ 212.3638968...,  135.        ,  414.6704871...]]),
     array([[ 400.1039651...,  405.        ,  191.7287234...],
           [ 271.0384615...,  135.        ,   13.5      ...],
           [ 495.4705323...,  405.        ,  119.7290874...],
           [ 580.7967033...,  405.        ,    6.6758241...],
           [ 190.1933701...,  135.        ,    7.4585635...],
           [ 473.7184115...,  405.        ,   10.2346570...],
           [ 135.4936014...,  135.        ,   20.2376599...],
           [  86.4689781...,  135.        ,   35.2281021...],
           [ 283.5396281...,  405.        ,  258.1775929...],
           [ 119.7044335...,  135.        ,  282.6354679...],
           [ 359.9532224...,  405.        ,  381.0031185...],
           [ 181.8271461...,  135.        ,  204.0661252...]]),
     array(1500),
     array(1500),
     0.3,
     0.3,
     {})
    """

    valid_experiment_results = (1, 2, 3, 4, 6, 8, 9, 11, 12)
    assert experiment in valid_experiment_results, (
        '"Breneman (1987)" experiment result must be one of "{0}"!'.format(
            valid_experiment_results))

    samples_luminance = [
        0.270,
        0.090,
        0.270,
        0.270,
        0.090,
        0.270,
        0.090,
        0.090,
        0.270,
        0.090,
        0.270,
        0.090,
    ]

    experiment_results = list(BRENEMAN_EXPERIMENTS[experiment])
    illuminant_chromaticities = experiment_results.pop(0)
    Y_r = Y_t = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES[experiment].Y
    B_r = B_t = 0.3

    XYZ_t, XYZ_r = xy_to_XYZ(
        np.hstack([
            Luv_uv_to_xy(illuminant_chromaticities[1:3]),
            np.full([2, 1], Y_r)
        ])) / Y_r

    xyY_cr, xyY_ct = [], []
    for i, experiment_result in enumerate(experiment_results):
        xyY_cr.append(
            np.hstack([
                Luv_uv_to_xy(experiment_result[2]), samples_luminance[i] * Y_r
            ]))
        xyY_ct.append(
            np.hstack([
                Luv_uv_to_xy(experiment_result[1]), samples_luminance[i] * Y_t
            ]))

    XYZ_cr = xyY_to_XYZ(xyY_cr)
    XYZ_ct = xyY_to_XYZ(xyY_ct)

    return CorrespondingColourDataset(experiment, XYZ_r, XYZ_t, XYZ_cr, XYZ_ct,
                                      Y_r, Y_t, B_r, B_t, {})


def corresponding_chromaticities_prediction_Fairchild1990(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *Fairchild (1990)*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer or CorrespondingColourDataset, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    References
    ----------
    :cite:`Breneman1987b`, :cite:`Fairchild1991a`, :cite:`Fairchild2013s`

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_Fairchild1990(2)
    >>> pr = [(p.uv_m, p.uv_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [(array([ 0.207,  0.486]), array([ 0.2089528...,  0.4724034...])),
     (array([ 0.449,  0.511]), array([ 0.4375652...,  0.5121030...])),
     (array([ 0.263,  0.505]), array([ 0.2621362...,  0.4972538...])),
     (array([ 0.322,  0.545]), array([ 0.3235312...,  0.5475665...])),
     (array([ 0.316,  0.537]), array([ 0.3151391...,  0.5398333...])),
     (array([ 0.265,  0.553]), array([ 0.2634745...,  0.5544335...])),
     (array([ 0.221,  0.538]), array([ 0.2211595...,  0.5324470...])),
     (array([ 0.135,  0.532]), array([ 0.1396949...,  0.5207234...])),
     (array([ 0.145,  0.472]), array([ 0.1512288...,  0.4533041...])),
     (array([ 0.163,  0.331]), array([ 0.1715691...,  0.3026264...])),
     (array([ 0.176,  0.431]), array([ 0.1825792...,  0.4077892...])),
     (array([ 0.244,  0.349]), array([ 0.2418905...,  0.3413401...]))]
    """

    experiment_results = (convert_experiment_results_Breneman1987(experiment)
                          if is_numeric(experiment) else experiment)

    with domain_range_scale(1):
        XYZ_t, XYZ_r = experiment_results.XYZ_t, experiment_results.XYZ_r
        xy_t, xy_r = XYZ_to_xy([XYZ_t, XYZ_r])

        uv_t = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_ct, xy_t), xy_t)
        uv_m = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_cr, xy_r), xy_r)

        Y_n = experiment_results.Y_t

        XYZ_1 = experiment_results.XYZ_ct
        XYZ_2 = chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_t, XYZ_r, Y_n)
        uv_p = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_r), xy_r)

        return tuple([
            CorrespondingChromaticitiesPrediction(experiment_results.name,
                                                  uv_t[i], uv_m[i], uv_p[i])
            for i in range(len(uv_t))
        ])


def corresponding_chromaticities_prediction_CIE1994(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CIE 1994*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer or CorrespondingColourDataset, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance. Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    References
    ----------
    :cite:`Breneman1987b`, :cite:`CIETC1-321994b`

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_CIE1994(2)
    >>> pr = [(p.uv_m, p.uv_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [(array([ 0.207,  0.486]), array([ 0.2273130...,  0.5267609...])),
     (array([ 0.449,  0.511]), array([ 0.4612181...,  0.5191849...])),
     (array([ 0.263,  0.505]), array([ 0.2872404...,  0.5306938...])),
     (array([ 0.322,  0.545]), array([ 0.3489822...,  0.5454398...])),
     (array([ 0.316,  0.537]), array([ 0.3371612...,  0.5421567...])),
     (array([ 0.265,  0.553]), array([ 0.2889416...,  0.5534074...])),
     (array([ 0.221,  0.538]), array([ 0.2412195...,  0.5464301...])),
     (array([ 0.135,  0.532]), array([ 0.1530344...,  0.5488239...])),
     (array([ 0.145,  0.472]), array([ 0.1568709...,  0.5258835...])),
     (array([ 0.163,  0.331]), array([ 0.1499762...,  0.4401747...])),
     (array([ 0.176,  0.431]), array([ 0.1876711...,  0.5039627...])),
     (array([ 0.244,  0.349]), array([ 0.2560012...,  0.4546263...]))]
    """

    experiment_results = (convert_experiment_results_Breneman1987(experiment)
                          if is_numeric(experiment) else experiment)

    with domain_range_scale(1):
        XYZ_t, XYZ_r = experiment_results.XYZ_t, experiment_results.XYZ_r
        xy_o1, xy_o2 = XYZ_to_xy([XYZ_t, XYZ_r])

        uv_t = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_ct, xy_o1), xy_o1)
        uv_m = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_cr, xy_o2), xy_o2)

        Y_r = experiment_results.B_r
        E_o1, E_o2 = experiment_results.Y_t, experiment_results.Y_r

        XYZ_1 = experiment_results.XYZ_ct
        XYZ_2 = chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_r, E_o1,
                                             E_o2)
        uv_p = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_o2), xy_o2)

        return tuple([
            CorrespondingChromaticitiesPrediction(experiment_results.name,
                                                  uv_t[i], uv_m[i], uv_p[i])
            for i in range(len(uv_t))
        ])


def corresponding_chromaticities_prediction_CMCCAT2000(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CMCCAT2000*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer or CorrespondingColourDataset, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    References
    ----------
    :cite:`Breneman1987b`, :cite:`Li2002a`, :cite:`Westland2012k`

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_CMCCAT2000(2)
    >>> pr = [(p.uv_m, p.uv_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [(array([ 0.207,  0.486]), array([ 0.2083210...,  0.4727168...])),
     (array([ 0.449,  0.511]), array([ 0.4459270...,  0.5077735...])),
     (array([ 0.263,  0.505]), array([ 0.2640262...,  0.4955361...])),
     (array([ 0.322,  0.545]), array([ 0.3316884...,  0.5431580...])),
     (array([ 0.316,  0.537]), array([ 0.3222624...,  0.5357624...])),
     (array([ 0.265,  0.553]), array([ 0.2710705...,  0.5501997...])),
     (array([ 0.221,  0.538]), array([ 0.2261826...,  0.5294740...])),
     (array([ 0.135,  0.532]), array([ 0.1439693...,  0.5190984...])),
     (array([ 0.145,  0.472]), array([ 0.1494835...,  0.4556760...])),
     (array([ 0.163,  0.331]), array([ 0.1563172...,  0.3164151...])),
     (array([ 0.176,  0.431]), array([ 0.1763199...,  0.4127589...])),
     (array([ 0.244,  0.349]), array([ 0.2287638...,  0.3499324...]))]
    """

    experiment_results = (convert_experiment_results_Breneman1987(experiment)
                          if is_numeric(experiment) else experiment)

    with domain_range_scale(1):
        XYZ_w, XYZ_wr = experiment_results.XYZ_t, experiment_results.XYZ_r
        xy_w, xy_wr = XYZ_to_xy([XYZ_w, XYZ_wr])

        uv_t = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_ct, xy_w), xy_w)
        uv_m = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_cr, xy_wr), xy_wr)

        L_A1 = experiment_results.Y_t
        L_A2 = experiment_results.Y_r

        XYZ_1 = experiment_results.XYZ_ct
        XYZ_2 = chromatic_adaptation_CMCCAT2000(XYZ_1, XYZ_w, XYZ_wr, L_A1,
                                                L_A2)
        uv_p = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)

        return tuple([
            CorrespondingChromaticitiesPrediction(experiment_results.name,
                                                  uv_t[i], uv_m[i], uv_p[i])
            for i in range(len(uv_t))
        ])


def corresponding_chromaticities_prediction_VonKries(experiment=1,
                                                     transform='CAT02'):
    """
    Returns the corresponding chromaticities prediction for *Von Kries*
    chromatic adaptation model using given transform.

    Parameters
    ----------
    experiment : integer or CorrespondingColourDataset, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance.
    transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    References
    ----------
    :cite:`Breneman1987b`, :cite:`Fairchild2013t`

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_VonKries(2, 'Bradford')
    >>> pr = [(p.uv_m, p.uv_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [(array([ 0.207,  0.486]), array([ 0.2082014...,  0.4722922...])),
     (array([ 0.449,  0.511]), array([ 0.4489102...,  0.5071602...])),
     (array([ 0.263,  0.505]), array([ 0.2643545...,  0.4959631...])),
     (array([ 0.322,  0.545]), array([ 0.3348730...,  0.5471220...])),
     (array([ 0.316,  0.537]), array([ 0.3248758...,  0.5390589...])),
     (array([ 0.265,  0.553]), array([ 0.2733105...,  0.5555028...])),
     (array([ 0.221,  0.538]), array([ 0.227148 ...,  0.5331318...)),
     (array([ 0.135,  0.532]), array([ 0.1442730...,  0.5226804...])),
     (array([ 0.145,  0.472]), array([ 0.1498745...,  0.4550785...])),
     (array([ 0.163,  0.331]), array([ 0.1564975...,  0.3148796...])),
     (array([ 0.176,  0.431]), array([ 0.1760593...,  0.4103772...])),
     (array([ 0.244,  0.349]), array([ 0.2259805...,  0.3465291...]))]
    """

    experiment_results = (convert_experiment_results_Breneman1987(experiment)
                          if is_numeric(experiment) else experiment)

    with domain_range_scale(1):
        XYZ_w, XYZ_wr = experiment_results.XYZ_t, experiment_results.XYZ_r
        xy_w, xy_wr = XYZ_to_xy([XYZ_w, XYZ_wr])

        uv_t = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_ct, xy_w), xy_w)
        uv_m = Luv_to_uv(XYZ_to_Luv(experiment_results.XYZ_cr, xy_wr), xy_wr)

        XYZ_1 = experiment_results.XYZ_ct
        XYZ_2 = chromatic_adaptation_VonKries(XYZ_1, XYZ_w, XYZ_wr, transform)
        uv_p = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)

        return tuple([
            CorrespondingChromaticitiesPrediction(experiment_results.name,
                                                  uv_t[i], uv_m[i], uv_p[i])
            for i in range(len(uv_t))
        ])


CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS = CaseInsensitiveMapping({
    'CIE 1994': corresponding_chromaticities_prediction_CIE1994,
    'CMCCAT2000': corresponding_chromaticities_prediction_CMCCAT2000,
    'Fairchild 1990': corresponding_chromaticities_prediction_Fairchild1990,
    'Von Kries': corresponding_chromaticities_prediction_VonKries
})
CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS.__doc__ = """
Aggregated corresponding chromaticities prediction models.

References
----------
:cite:`Breneman1987b`, :cite:`CIETC1-321994b`, :cite:`Fairchild1991a`,
:cite:`Fairchild2013s`, :cite:`Fairchild2013t`, :cite:`Li2002a`,
:cite:`Westland2012k`

CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS : CaseInsensitiveMapping
    **{'CIE 1994', 'CMCCAT2000', 'Fairchild 1990', 'Von Kries'}**

Aliases:

-   'vonkries': 'Von Kries'
"""
CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS['vonkries'] = (
    CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS['Von Kries'])


def corresponding_chromaticities_prediction(experiment=1,
                                            model='Von Kries',
                                            **kwargs):
    """
    Returns the corresponding chromaticities prediction for given chromatic
    adaptation model.

    Parameters
    ----------
    experiment : integer or CorrespondingColourDataset, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance.
    model : unicode, optional
        **{'Von Kries', 'CIE 1994', 'CMCCAT2000', 'Fairchild 1990'}**,
        Chromatic adaptation model.

    Other Parameters
    ----------------
    transform : unicode, optional
        {:func:`colour.corresponding.\
corresponding_chromaticities_prediction_VonKries`},
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    References
    ----------
    :cite:`Breneman1987b`, :cite:`CIETC1-321994b`, :cite:`Fairchild1991a`,
    :cite:`Fairchild2013s`, :cite:`Fairchild2013t`, :cite:`Li2002a`,
    :cite:`Westland2012k`

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction(2, 'CMCCAT2000')
    >>> pr = [(p.uv_m, p.uv_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.2083210..., 0.4727168...)),
     ((0.449, 0.511), (0.4459270..., 0.5077735...)),
     ((0.263, 0.505), (0.2640262..., 0.4955361...)),
     ((0.322, 0.545), (0.3316884..., 0.5431580...)),
     ((0.316, 0.537), (0.3222624..., 0.5357624...)),
     ((0.265, 0.553), (0.2710705..., 0.5501997...)),
     ((0.221, 0.538), (0.2261826..., 0.5294740...)),
     ((0.135, 0.532), (0.1439693..., 0.5190984...)),
     ((0.145, 0.472), (0.1494835..., 0.4556760...)),
     ((0.163, 0.331), (0.1563172..., 0.3164151...)),
     ((0.176, 0.431), (0.1763199..., 0.4127589...)),
     ((0.244, 0.349), (0.2287638..., 0.3499324...))]
    """

    function = CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS[model]

    return function(experiment, **filter_kwargs(function, **kwargs))
