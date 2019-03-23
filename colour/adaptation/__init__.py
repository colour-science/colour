# -*- coding: utf-8 -*-
"""
References
----------
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

from __future__ import absolute_import

from colour.utilities import (CaseInsensitiveMapping, filter_kwargs,
                              get_domain_range_scale, as_float_array)

from .dataset import *  # noqa
from . import dataset
from .vonkries import (chromatic_adaptation_matrix_VonKries,
                       chromatic_adaptation_VonKries)
from .fairchild1990 import chromatic_adaptation_Fairchild1990
from .cmccat2000 import (
    CMCCAT2000_InductionFactors, CMCCAT2000_VIEWING_CONDITIONS,
    chromatic_adaptation_forward_CMCCAT2000,
    chromatic_adaptation_reverse_CMCCAT2000, chromatic_adaptation_CMCCAT2000)
from .cie1994 import chromatic_adaptation_CIE1994

__all__ = []
__all__ += dataset.__all__
__all__ += [
    'chromatic_adaptation_matrix_VonKries', 'chromatic_adaptation_VonKries'
]
__all__ += ['chromatic_adaptation_Fairchild1990']
__all__ += [
    'CMCCAT2000_InductionFactors', 'CMCCAT2000_VIEWING_CONDITIONS',
    'chromatic_adaptation_forward_CMCCAT2000',
    'chromatic_adaptation_reverse_CMCCAT2000',
    'chromatic_adaptation_CMCCAT2000'
]
__all__ += ['chromatic_adaptation_CIE1994']

CHROMATIC_ADAPTATION_METHODS = CaseInsensitiveMapping({
    'CIE 1994': chromatic_adaptation_CIE1994,
    'CMCCAT2000': chromatic_adaptation_CMCCAT2000,
    'Fairchild 1990': chromatic_adaptation_Fairchild1990,
    'Von Kries': chromatic_adaptation_VonKries,
})
CHROMATIC_ADAPTATION_METHODS.__doc__ = """
Supported chromatic adaptation methods.

References
----------
:cite:`CIETC1-321994b`, :cite:`Fairchild1991a`, :cite:`Fairchild2013s`,
:cite:`Fairchild2013t`, :cite:`Li2002a`, :cite:`Westland2012k`

CHROMATIC_ADAPTATION_METHODS : CaseInsensitiveMapping
    **{'CIE 1994', 'CMCCAT2000', 'Fairchild 1990', 'Von Kries'}**
"""


def chromatic_adaptation(XYZ, XYZ_w, XYZ_wr, method='Von Kries', **kwargs):
    """
    Adapts given stimulus from test viewing conditions to reference viewing
    conditions.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of stimulus to adapt.
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* tristimulus values of the whitepoint.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* tristimulus values of the
        whitepoint.
    method : unicode, optional
        **{'Von Kries', 'CIE 1994', 'CMCCAT2000', 'Fairchild 1990'}**,
        Computation method.

    Other Parameters
    ----------------
    E_o1 : numeric
        {:func:`colour.adaptation.chromatic_adaptation_CIE1994`},
        Test illuminance :math:`E_{o1}` in :math:`cd/m^2`.
    E_o2 : numeric
        {:func:`colour.adaptation.chromatic_adaptation_CIE1994`},
        Reference illuminance :math:`E_{o2}` in :math:`cd/m^2`.
    L_A1 : numeric or array_like
        {:func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`},
        Luminance of test adapting field :math:`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric or array_like
        {:func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`},
        Luminance of reference adapting field :math:`L_{A2}` in :math:`cd/m^2`.
    Y_n : numeric or array_like
        {:func:`colour.adaptation.chromatic_adaptation_Fairchild1990`},
        Luminance :math:`Y_n` of test adapting stimulus in :math:`cd/m^2`.
    Y_o : numeric
        {:func:`colour.adaptation.chromatic_adaptation_CIE1994`},
        Luminance factor :math:`Y_o` of achromatic background normalised to
        domain [0.18, 1] in **'Reference'** domain-range scale.
    direction : unicode, optional
        {:func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`},
        **{'Forward', 'Reverse'}**,
        Chromatic adaptation direction.
    discount_illuminant : bool, optional
        {:func:`colour.adaptation.chromatic_adaptation_Fairchild1990`},
        Truth value indicating if the illuminant should be discounted.
    n : numeric, optional
        {:func:`colour.adaptation.chromatic_adaptation_CIE1994`},
        Noise component in fundamental primary system.
    surround : CMCCAT2000_InductionFactors, optional
        {:func:`colour.adaptation.chromatic_adaptation_CMCCAT2000`},
        Surround viewing conditions induction factors.
    transform : unicode, optional
        {:func:`colour.adaptation.chromatic_adaptation_VonKries`},
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    ndarray
        *CIE XYZ_c* tristimulus values of the stimulus corresponding colour.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_w``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``Y_o``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_c``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-321994b`, :cite:`Fairchild1991a`, :cite:`Fairchild2013s`,
    :cite:`Fairchild2013t`, :cite:`Li2002a`, :cite:`Westland2012k`

    Examples
    --------

    *Von Kries* chromatic adaptation:

    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> chromatic_adaptation(XYZ, XYZ_w, XYZ_wr)
    ... # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])

    *CIE 1994* chromatic adaptation, requires extra *kwargs*:

    >>> XYZ = np.array([0.2800, 0.2126, 0.0527])
    >>> XYZ_w = np.array([1.09867452, 1.00000000, 0.35591556])
    >>> XYZ_wr = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> Y_o = 0.20
    >>> E_o = 1000
    >>> chromatic_adaptation(
    ...     XYZ, XYZ_w, XYZ_wr, method='CIE 1994', Y_o=Y_o, E_o1=E_o, E_o2=E_o)
    ... # doctest: +ELLIPSIS
    array([ 0.2403379...,  0.2115621...,  0.1764301...])

    *CMCCAT2000* chromatic adaptation, requires extra *kwargs*:

    >>> XYZ = np.array([0.2248, 0.2274, 0.0854])
    >>> XYZ_w = np.array([1.1115, 1.0000, 0.3520])
    >>> XYZ_wr = np.array([0.9481, 1.0000, 1.0730])
    >>> L_A = 200
    >>> chromatic_adaptation(
    ...     XYZ, XYZ_w, XYZ_wr, method='CMCCAT2000', L_A1=L_A, L_A2=L_A)
    ... # doctest: +ELLIPSIS
    array([ 0.1952698...,  0.2306834...,  0.2497175...])

    *Fairchild (1990)* chromatic adaptation, requires extra *kwargs*:

    >>> XYZ = np.array([0.1953, 0.2307, 0.2497])
    >>> Y_n = 200
    >>> chromatic_adaptation(
    ...     XYZ, XYZ_w, XYZ_wr, method='Fairchild 1990', Y_n=Y_n)
    ... # doctest: +ELLIPSIS
    array([ 0.2332526...,  0.2332455...,  0.7611593...])
    """

    function = CHROMATIC_ADAPTATION_METHODS[method]

    domain_range_reference = get_domain_range_scale() == 'reference'
    domain_100 = (chromatic_adaptation_CIE1994,
                  chromatic_adaptation_CMCCAT2000,
                  chromatic_adaptation_Fairchild1990)

    if function in domain_100 and domain_range_reference:
        XYZ = as_float_array(XYZ) * 100
        XYZ_w = as_float_array(XYZ_w) * 100
        XYZ_wr = as_float_array(XYZ_wr) * 100
        if kwargs.get('Y_o'):
            kwargs['Y_o'] = kwargs['Y_o'] * 100

    kwargs.update({'XYZ_w': XYZ_w, 'XYZ_wr': XYZ_wr})

    if function is chromatic_adaptation_CIE1994:
        from colour import XYZ_to_xy

        kwargs.update({'xy_o1': XYZ_to_xy(XYZ_w), 'xy_o2': XYZ_to_xy(XYZ_wr)})

    elif function is chromatic_adaptation_Fairchild1990:
        kwargs.update({'XYZ_n': XYZ_w, 'XYZ_r': XYZ_wr})

    XYZ_c = function(XYZ, **filter_kwargs(function, **kwargs))

    if function in domain_100 and domain_range_reference:
        XYZ_c /= 100

    return XYZ_c


__all__ += ['CHROMATIC_ADAPTATION_METHODS', 'chromatic_adaptation']
