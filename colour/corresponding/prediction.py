#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corresponding Chromaticities Prediction
=======================================

Defines objects to compute corresponding chromaticities prediction.

See Also
--------
`Corresponding Chromaticities Prediction IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/corresponding/prediction.ipynb>`_

References
----------
.. [1]  Breneman, E. J. (1987). Corresponding chromaticities for different
        states of adaptation to complex visual fields. JOSA A, 4(6). Retrieved
        from http://www.opticsinfobase.org/josaa/\
fulltext.cfm?uri=josaa-4-6-1115&id=2783
"""

from __future__ import division, unicode_literals

from collections import namedtuple

from colour.adaptation import (
    chromatic_adaptation_CIE1994,
    chromatic_adaptation_CMCCAT2000,
    chromatic_adaptation_Fairchild1990,
    chromatic_adaptation_VonKries)
from colour.corresponding import (
    BRENEMAN_EXPERIMENTS,
    BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES)
from colour.models import (
    Luv_to_uv,
    Luv_uv_to_xy,
    XYZ_to_Luv,
    XYZ_to_xy,
    xy_to_XYZ)
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CorrespondingChromaticitiesPrediction',
           'corresponding_chromaticities_prediction_CIE1994',
           'corresponding_chromaticities_prediction_CMCCAT2000',
           'corresponding_chromaticities_prediction_Fairchild1990',
           'corresponding_chromaticities_prediction_VonKries',
           'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS']


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


def corresponding_chromaticities_prediction_CIE1994(experiment=1, **kwargs):
    """
    Returns the corresponding chromaticities prediction for CIE 1994
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        Breneman (1987) experiment number.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_CIE1994(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.21339093279517196, 0.49397945742298016)),
     ((0.449, 0.511), (0.4450345313098153, 0.5120939085633327)),
     ((0.263, 0.505), (0.26932620724691858, 0.50832124608390727)),
     ((0.322, 0.545), (0.33085939370840811, 0.54439408389253441)),
     ((0.316, 0.537), (0.3225195584183046, 0.53778269440789594)),
     ((0.265, 0.553), (0.2709737181087471, 0.5513666373694861)),
     ((0.221, 0.538), (0.22807869730753863, 0.53515923458385406)),
     ((0.135, 0.532), (0.14394366662060523, 0.53035769204585748)),
     ((0.145, 0.472), (0.15007438031976222, 0.48428958620888679)),
     ((0.163, 0.331), (0.15599555781959967, 0.37723798698131394)),
     ((0.176, 0.431), (0.18063180902005657, 0.45184759430042898)),
     ((0.244, 0.349), (0.24544456656434688, 0.40180048388092021))]
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
        XYZ_2 = chromatic_adaptation_CIE1994(
            XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_o2), xy_o2)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_CMCCAT2000(experiment=1, **kwargs):
    """
    Returns the corresponding chromaticities prediction for CMCCAT2000
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        Breneman (1987) experiment number.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_CMCCAT2000(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.20832101929657834, 0.47271680534693694)),
     ((0.449, 0.511), (0.44592707020371486, 0.50777351504395707)),
     ((0.263, 0.505), (0.26402624712986333, 0.4955361681706304)),
     ((0.322, 0.545), (0.33168840090358015, 0.54315801981008516)),
     ((0.316, 0.537), (0.32226245779851387, 0.53576245377085929)),
     ((0.265, 0.553), (0.27107058097430181, 0.5501997842556422)),
     ((0.221, 0.538), (0.22618269421847523, 0.52947407170848704)),
     ((0.135, 0.532), (0.14396930475660724, 0.51909841743126817)),
     ((0.145, 0.472), (0.14948357434418671, 0.45567605010224305)),
     ((0.163, 0.331), (0.15631720730028753, 0.31641514460738623)),
     ((0.176, 0.431), (0.17631993066748047, 0.41275893424542082)),
     ((0.244, 0.349), (0.22876382018951744, 0.3499324084859976))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t)) * 100
    XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m)) * 100
    xy_wr = XYZ_to_xy(XYZ_wr)
    L_A1 = L_A2 = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.get(
        experiment).Y

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t)) * 100
        XYZ_2 = chromatic_adaptation_CMCCAT2000(
            XYZ_1, XYZ_w, XYZ_wr, L_A1, L_A2)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_Fairchild1990(experiment=1,
                                                          **kwargs):
    """
    Returns the corresponding chromaticities prediction for Fairchild (1990)
    chromatic adaptation model.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        Breneman (1987) experiment number.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_Fairchild1990(2)
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.2089528677990308, 0.47240345174230519)),
     ((0.449, 0.511), (0.43756528098582792, 0.51210303139041924)),
     ((0.263, 0.505), (0.26213623665658092, 0.49725385033264224)),
     ((0.322, 0.545), (0.3235312762825191, 0.54756652922585702)),
     ((0.316, 0.537), (0.3151390992740366, 0.53983332031574016)),
     ((0.265, 0.553), (0.26347459238415272, 0.55443357809543037)),
     ((0.221, 0.538), (0.22115956537655593, 0.53244703908294599)),
     ((0.135, 0.532), (0.13969494108553854, 0.52072342107668024)),
     ((0.145, 0.472), (0.1512288710743511, 0.45330415352961834)),
     ((0.163, 0.331), (0.17156913711903982, 0.30262647410866889)),
     ((0.176, 0.431), (0.18257922398137369, 0.40778921192793854)),
     ((0.244, 0.349), (0.24189049501108895, 0.34134012046930529))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_n = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t)) * 100
    XYZ_r = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m)) * 100
    xy_r = XYZ_to_xy(XYZ_r)
    Y_n = BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.get(experiment).Y

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t)) * 100
        XYZ_2 = chromatic_adaptation_Fairchild1990(
            XYZ_1, XYZ_n, XYZ_r, Y_n)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_r), xy_r)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


def corresponding_chromaticities_prediction_VonKries(experiment=1,
                                                     transform='CAT02'):
    """
    Returns the corresponding chromaticities prediction for Von Kries
    chromatic adaptation model using given transform.

    Parameters
    ----------
    experiment : integer, optional
        {1, 2, 3, 4, 6, 8, 9, 11, 12}
        Breneman (1987) experiment number.
    transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild, 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    tuple
        Corresponding chromaticities prediction.

    Examples
    --------
    >>> from pprint import pprint
    >>> pr = corresponding_chromaticities_prediction_VonKries(2, 'Bradford')
    >>> pr = [(p.uvp_m, p.uvp_p) for p in pr]
    >>> pprint(pr)  # doctest: +SKIP
    [((0.207, 0.486), (0.20820148430638033, 0.47229226819364528)),
     ((0.449, 0.511), (0.44891022948064191, 0.50716028901449561)),
     ((0.263, 0.505), (0.26435459360846608, 0.49596314494922683)),
     ((0.322, 0.545), (0.33487309037107632, 0.54712207251983425)),
     ((0.316, 0.537), (0.32487581236911361, 0.53905899356457776)),
     ((0.265, 0.553), (0.27331050571632376, 0.55550280647813977)),
     ((0.221, 0.538), (0.22714800102072819, 0.53313179748041983)),
     ((0.135, 0.532), (0.14427303768336433, 0.52268044497913713)),
     ((0.145, 0.472), (0.14987451889726533, 0.45507852741116867)),
     ((0.163, 0.331), (0.15649757464732098, 0.31487959772753954)),
     ((0.176, 0.431), (0.17605936460371163, 0.41037722722471409)),
     ((0.244, 0.349), (0.22598059059292835, 0.34652914678030416))]
    """

    experiment_results = list(BRENEMAN_EXPERIMENTS.get(experiment))

    illuminants = experiment_results.pop(0)
    XYZ_w = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_t))
    XYZ_wr = xy_to_XYZ(Luv_uv_to_xy(illuminants.uvp_m))
    xy_wr = XYZ_to_xy(XYZ_wr)

    prediction = []
    for result in experiment_results:
        XYZ_1 = xy_to_XYZ(Luv_uv_to_xy(result.uvp_t))
        XYZ_2 = chromatic_adaptation_VonKries(XYZ_1, XYZ_w, XYZ_wr, transform)
        uvp = Luv_to_uv(XYZ_to_Luv(XYZ_2, xy_wr), xy_wr)
        prediction.append(CorrespondingChromaticitiesPrediction(
            result.name,
            result.uvp_t,
            result.uvp_m,
            uvp))

    return tuple(prediction)


CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS = CaseInsensitiveMapping(
    {'CIE 1994': corresponding_chromaticities_prediction_CIE1994,
     'CMCCAT2000': corresponding_chromaticities_prediction_CMCCAT2000,
     'Fairchild 1990': corresponding_chromaticities_prediction_Fairchild1990,
     'Von Kries': corresponding_chromaticities_prediction_VonKries})

"""
Aggregated corresponding chromaticities prediction models.

CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS : CaseInsensitiveMapping
    **{'CIE 1994', 'CMCCAT2000', 'Fairchild 1990', 'Von Kries'}**

Aliases:

-   'vonkries': 'Von Kries'
"""
CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS['vonkries'] = (
    CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS['Von Kries'])
