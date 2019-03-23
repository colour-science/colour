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
-   :cite:`Westland2012k` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT2000. In Computational Colour Science Using MATLAB
    (2nd ed., pp. 83-86). ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

from collections import namedtuple

from colour.adaptation import (
    chromatic_adaptation_CIE1994, chromatic_adaptation_CMCCAT2000,
    chromatic_adaptation_Fairchild1990, chromatic_adaptation_VonKries)
from colour.corresponding import (
    BRENEMAN_EXPERIMENTS, BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES)
from colour.models import (Luv_to_uv, Luv_uv_to_xy, XYZ_to_Luv, XYZ_to_xy,
                           xy_to_XYZ)
from colour.utilities import (CaseInsensitiveMapping, domain_range_scale,
                              filter_kwargs)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'CorrespondingChromaticitiesPrediction',
    'corresponding_chromaticities_prediction_Fairchild1990',
    'corresponding_chromaticities_prediction_CIE1994',
    'corresponding_chromaticities_prediction_CMCCAT2000',
    'corresponding_chromaticities_prediction_VonKries',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'corresponding_chromaticities_prediction'
]


class CorrespondingChromaticitiesPrediction(
        namedtuple('CorrespondingChromaticitiesPrediction',
                   ('name', 'uvp_t', 'uvp_m', 'uvp_p'))):
    """
    Defines a chromatic adaptation model prediction.

    Parameters
    ----------
    name : unicode
        Test colour name.
    uvp_t : numeric
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    uvp_m : array_like, (2,)
        Chromaticity coordinates :math:`uv_m^p` of matching colour.
    uvp_p : array_like, (2,)
        Chromaticity coordinates :math:`uv_p^p` of predicted colour.
    """


def corresponding_chromaticities_prediction_Fairchild1990(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *Fairchild (1990)*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.

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
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.2089528..., 0.4724034...)),
     ((0.449, 0.511), (0.4375652..., 0.5121030...)),
     ((0.263, 0.505), (0.2621362..., 0.4972538...)),
     ((0.322, 0.545), (0.3235312..., 0.5475665...)),
     ((0.316, 0.537), (0.3151390..., 0.5398333...)),
     ((0.265, 0.553), (0.2634745..., 0.5544335...)),
     ((0.221, 0.538), (0.2211595..., 0.5324470...)),
     ((0.135, 0.532), (0.1396949..., 0.5207234...)),
     ((0.145, 0.472), (0.1512288..., 0.4533041...)),
     ((0.163, 0.331), (0.1715691..., 0.3026264...)),
     ((0.176, 0.431), (0.1825792..., 0.4077892...)),
     ((0.244, 0.349), (0.2418904..., 0.3413401...))]
    """

    with domain_range_scale(1):
        experiment_results = list(BRENEMAN_EXPERIMENTS[experiment])

        illuminants = experiment_results.pop(0)
        XYZ_n = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t))
        XYZ_r = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m))
        xy_r = XYZ_to_xy(XYZ_r)
        Y_n = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES[experiment].Y

        prediction = []
        for result in experiment_results:
            XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
            XYZ_2 = chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r,
                                                       Y_n)
            uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_r), xy_r)
            prediction.append(
                CorrespondingChromaticitiesPrediction(
                    result.name, result.uvp_t, result.uvp_m, uvp))

        return tuple(prediction)


def corresponding_chromaticities_prediction_CIE1994(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CIE 1994*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.

    Returns
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
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.2133909..., 0.4939794...)),
     ((0.449, 0.511), (0.4450345..., 0.5120939...)),
     ((0.263, 0.505), (0.2693262..., 0.5083212...)),
     ((0.322, 0.545), (0.3308593..., 0.5443940...)),
     ((0.316, 0.537), (0.3225195..., 0.5377826...)),
     ((0.265, 0.553), (0.2709737..., 0.5513666...)),
     ((0.221, 0.538), (0.2280786..., 0.5351592...)),
     ((0.135, 0.532), (0.1439436..., 0.5303576...)),
     ((0.145, 0.472), (0.1500743..., 0.4842895...)),
     ((0.163, 0.331), (0.1559955..., 0.3772379...)),
     ((0.176, 0.431), (0.1806318..., 0.4518475...)),
     ((0.244, 0.349), (0.2454445..., 0.4018004...))]
    """

    with domain_range_scale(1):
        experiment_results = list(BRENEMAN_EXPERIMENTS[experiment])

        illuminants = experiment_results.pop(0)
        xy_o1 = Luv_uv_to_xy(illuminants.uvp_t)
        xy_o2 = Luv_uv_to_xy(illuminants.uvp_m)
        # :math:`Y_o` is set to an arbitrary value normalised to domain
        # [18, 100].
        Y_o = 0.18
        E_o1 = E_o2 = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES[
            experiment].Y

        prediction = []
        for result in experiment_results:
            XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
            XYZ_2 = chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o,
                                                 E_o1, E_o2)
            uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_o2), xy_o2)
            prediction.append(
                CorrespondingChromaticitiesPrediction(
                    result.name, result.uvp_t, result.uvp_m, uvp))

        return tuple(prediction)


def corresponding_chromaticities_prediction_CMCCAT2000(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CMCCAT2000*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.

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
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
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

    with domain_range_scale(1):
        experiment_results = list(BRENEMAN_EXPERIMENTS[experiment])

        illuminants = experiment_results.pop(0)
        XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t))
        XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m))
        xy_wr = XYZ_to_xy(XYZ_wr)
        L_A1 = L_A2 = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES[
            experiment].Y

        prediction = []
        for result in experiment_results:
            XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
            XYZ_2 = chromatic_adaptation_CMCCAT2000(XYZ_1, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2)
            uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)
            prediction.append(
                CorrespondingChromaticitiesPrediction(
                    result.name, result.uvp_t, result.uvp_m, uvp))

        return tuple(prediction)


def corresponding_chromaticities_prediction_VonKries(experiment=1,
                                                     transform='CAT02'):
    """
    Returns the corresponding chromaticities prediction for *Von Kries*
    chromatic adaptation model using given transform.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.
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
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.2082014..., 0.4722922...)),
     ((0.449, 0.511), (0.4489102..., 0.5071602...)),
     ((0.263, 0.505), (0.2643545..., 0.4959631...)),
     ((0.322, 0.545), (0.3348730..., 0.5471220...)),
     ((0.316, 0.537), (0.3248758..., 0.5390589...)),
     ((0.265, 0.553), (0.2733105..., 0.5555028...)),
     ((0.221, 0.538), (0.2271480..., 0.5331317...)),
     ((0.135, 0.532), (0.1442730..., 0.5226804...)),
     ((0.145, 0.472), (0.1498745..., 0.4550785...)),
     ((0.163, 0.331), (0.1564975..., 0.3148795...)),
     ((0.176, 0.431), (0.1760593..., 0.4103772...)),
     ((0.244, 0.349), (0.2259805..., 0.3465291...))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS[experiment])

    illuminants = experiment_results.pop(0)
    XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t))
    XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m))
    xy_wr = XYZ_to_xy(XYZ_wr)

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
        XYZ_2 = chromatic_adaptation_VonKries(XYZ_1, XYZ_w, XYZ_wr, transform)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)
        prediction.append(
            CorrespondingChromaticitiesPrediction(result.name, result.uvp_t,
                                                  result.uvp_m, uvp))

    return tuple(prediction)


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
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Breneman (1987)* experiment number.
    model : unicode, optional
        **{'Von Kries', 'CIE 1994', 'CMCCAT2000', 'Fairchild 1990'}**,
        Chromatic adaptation model.

    Other Parameters
    ----------------
    transform : unicode, optional
        {:func:`colour.corresponding.corresponding_chromaticities_prediction_VonKries`},
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
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
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
