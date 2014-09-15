#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corresponding Chromaticities Prediction
=======================================

Defines objects to compute corresponding chromaticities prediction.

See Also
--------
`Corresponding Chromaticities IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/corresponding/corresponding_chromaticities.ipynb>`_  # noqa

References
----------
.. [1]  **Edwin J. Breneman**, *Corresponding chromaticities for different
        states of adaptation to complex visual fields*,
        *JOSA A, Vol. 4, Issue 6, pp. 1115-1129 (1987)*,
        DOI: http://dx.doi.org/10.1364/JOSAA.4.001115
"""

from __future__ import division, unicode_literals

from collections import namedtuple

from colour.adaptation import (
    chromatic_adaptation_vonkries,
    chromatic_adaptation_cie1994,
    chromatic_adaptation_CMCCAT2000,
    chromatic_adaptation_fairchild1990)
from colour.corresponding import (
    BRENEMAN_EXPERIMENTS,
    BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES)
from colour.models import (
    XYZ_to_xy,
    XYZ_to_Luv,
    Luv_to_uv,
    Luv_uv_to_xy,
    xy_to_XYZ)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CorrespondingChromaticitiesPrediction',
           'corresponding_chromaticities_prediction_vonkries',
           'corresponding_chromaticities_prediction_cie1994',
           'corresponding_chromaticities_prediction_CMCCAT2000',
           'corresponding_chromaticities_prediction_fairchild1990']


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


def corresponding_chromaticities_prediction_vonkries(experiment=1,
                                                     transform='CAT02'):
    """
    Returns the corresponding chromaticities prediction for *Von Kries*
    chromatic adaptation model using given transform.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Edwin J. Breneman* experiment number.
    transform : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
        Chromatic adaptation transform.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_vonkries(2, 'Bradford')
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [((0.20..., 0.48...), (0.2082014..., 0.4722922...)),
     ((0.44..., 0.51...), (0.4489102..., 0.5071602...)),
     ((0.26..., 0.50...), (0.2643545..., 0.4959631...)),
     ((0.32..., 0.54...), (0.3348730..., 0.5471220...)),
     ((0.31..., 0.53...), (0.3248758..., 0.5390589...)),
     ((0.26..., 0.55...), (0.2733105..., 0.5555028...)),
     ((0.22..., 0.53...), (0.2271480..., 0.5331317...)),
     ((0.13..., 0.53...), (0.1442730..., 0.5226804...)),
     ((0.14..., 0.47...), (0.1498745..., 0.4550785...)),
     ((0.16..., 0.33...), (0.1564975..., 0.3148795...)),
     ((0.17..., 0.43...), (0.1760593..., 0.4103772...)),
     ((0.24..., 0.34...), (0.2259805..., 0.3465291...))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t))
    XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m))
    xy_wr = XYZ_to_xy(XYZ_wr)

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
        XYZ_2 = chromatic_adaptation_vonkries(XYZ_1, XYZ_w, XYZ_wr, transform)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_cie1994(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CIE 1994*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Edwin J. Breneman* experiment number.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_cie1994(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [((0.20..., 0.48...), (0.2133909..., 0.4939794...)),
     ((0.44..., 0.51...), (0.4450345..., 0.5120939...)),
     ((0.26..., 0.50...), (0.2693262..., 0.5083212...)),
     ((0.32..., 0.54...), (0.3308593..., 0.5443940...)),
     ((0.31..., 0.53...), (0.3225195..., 0.5377826...)),
     ((0.26..., 0.55...), (0.2709737..., 0.5513666...)),
     ((0.22..., 0.53...), (0.2280786..., 0.5351592...)),
     ((0.13..., 0.53...), (0.1439436..., 0.5303576...)),
     ((0.14..., 0.47...), (0.1500743..., 0.4842895...)),
     ((0.16..., 0.33...), (0.1559955..., 0.3772379...)),
     ((0.17..., 0.43...), (0.1806318..., 0.4518475...)),
     ((0.24..., 0.34...), (0.2454445..., 0.4018004...))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    xy_o1 = Luv_uv_to_xy(illuminants.uvp_t)
    xy_o2 = Luv_uv_to_xy(illuminants.uvp_m)
    # :math:`Y_o` is set to an arbitrary value in domain [18, 100].
    Y_o = 18
    E_o1 = E_o2 = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.get(
        experiment).Y

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t)) * 100
        XYZ_2 = chromatic_adaptation_cie1994(
            XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_o2), xy_o2)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_CMCCAT2000(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *CMCCAT2000*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Edwin J. Breneman* experiment number.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_CMCCAT2000(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [((0.20..., 0.48...), (0.2083210..., 0.4727168...)),
     ((0.44..., 0.51...), (0.4459270..., 0.5077735...)),
     ((0.26..., 0.50...), (0.2640262..., 0.4955361...)),
     ((0.32..., 0.54...), (0.3316884..., 0.5431580...)),
     ((0.31..., 0.53...), (0.3222624..., 0.5357624...)),
     ((0.26..., 0.55...), (0.2710705..., 0.5501997...)),
     ((0.22..., 0.53...), (0.2261826..., 0.5294740...)),
     ((0.13..., 0.53...), (0.1439693..., 0.5190984...)),
     ((0.14..., 0.47...), (0.1494835..., 0.4556760...)),
     ((0.16..., 0.33...), (0.1563172..., 0.3164151...)),
     ((0.17..., 0.43...), (0.1763199..., 0.4127589...)),
     ((0.24..., 0.34...), (0.2287638..., 0.3499324...))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t)) * 100
    XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m)) * 100
    L_A1 = L_A2 = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.get(
        experiment).Y

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t)) * 100
        XYZ_2 = chromatic_adaptation_CMCCAT2000(
            XYZ_1, XYZ_w, XYZ_wr, L_A1, L_A2)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, XYZ_wr), XYZ_wr)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_fairchild1990(experiment=1):
    """
    Returns the corresponding chromaticities prediction for *Fairchild (1990)*
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        *Edwin J. Breneman* experiment number.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_fairchild1990(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +ELLIPSIS
    [((0.20..., 0.48...), (0.2089528..., 0.4724034...)),
     ((0.44..., 0.51...), (0.4375652..., 0.5121030...)),
     ((0.26..., 0.50...), (0.2621362..., 0.4972538...)),
     ((0.32..., 0.54...), (0.3235312..., 0.5475665...)),
     ((0.31..., 0.53...), (0.3151390..., 0.5398333...)),
     ((0.26..., 0.55...), (0.2634745..., 0.5544335...)),
     ((0.22..., 0.53...), (0.2211595..., 0.5324470...)),
     ((0.13..., 0.53...), (0.1396949..., 0.5207234...)),
     ((0.14..., 0.47...), (0.1512288..., 0.4533041...)),
     ((0.16..., 0.33...), (0.1715691..., 0.3026264...)),
     ((0.17..., 0.43...), (0.1825792..., 0.4077892...)),
     ((0.24..., 0.34...), (0.2418904..., 0.3413401...))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_n = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t)) * 100
    XYZ_r = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m)) * 100
    Y_n = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.get(experiment).Y

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t)) * 100
        XYZ_2 = chromatic_adaptation_fairchild1990(
            XYZ_1, XYZ_n, XYZ_r, Y_n)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, XYZ_r), XYZ_r)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)
