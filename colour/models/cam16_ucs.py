# -*- coding: utf-8 -*-
"""
CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces - Li et al. (2017)
===================================================================

Defines the *Li, Li, Wang, Zu, Luo, Cui, Melgosa, Brill and Pointer (2017)*
*CAM16-LCD*, *CAM16-SCD*, and *CAM16-UCS* colourspaces transformations:

-   :func:`colour.JMh_CAM16_to_CAM16LCD`
-   :func:`colour.CAM16LCD_to_JMh_CAM16`
-   :func:`colour.JMh_CAM16_to_CAM16SCD`
-   :func:`colour.CAM16SCD_to_JMh_CAM16`
-   :func:`colour.JMh_CAM16_to_CAM16UCS`
-   :func:`colour.CAM16UCS_to_JMh_CAM16`

See Also
--------
`CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cam16_ucs.ipynb>`_

References
----------
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Pointer, M. (2017). Comprehensive color solutions: CAM16, CAT16, and
    CAM16-UCS. Color Research & Application, 42(6), 703-718.
    doi:10.1002/col.22131
"""

from __future__ import division, unicode_literals

import re
from functools import partial

from colour.models.cam02_ucs import (
    COEFFICIENTS_UCS_LUO2006, JMh_CIECAM02_to_UCS_Luo2006,
    UCS_Luo2006_to_JMh_CIECAM02, JMh_CIECAM02_to_CAM02LCD,
    CAM02LCD_to_JMh_CIECAM02, JMh_CIECAM02_to_CAM02SCD,
    CAM02SCD_to_JMh_CIECAM02, JMh_CIECAM02_to_CAM02UCS,
    CAM02UCS_to_JMh_CIECAM02)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'JMh_CAM16_to_UCS_Li2017', 'UCS_Li2017_to_JMh_CAM16',
    'JMh_CAM16_to_CAM16LCD', 'CAM16LCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16SCD',
    'CAM16SCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16UCS', 'CAM16UCS_to_JMh_CAM16'
]


def _UCS_Luo2006_callable_to_UCS_Li2017_docstring(callable_):
    """
    Converts given *Luo et al. (2006)* callable docstring to
    *Li et al. (2017)* docstring.

    Parameters
    ----------
    callable_ : callable
        Callable to use the docstring from.

    Returns
    -------
    unicode
        Docstring.
    """

    docstring = callable_.__doc__
    docstring = docstring.replace('Luo et al. (2006)', 'Li et al. (2017)')
    docstring = docstring.replace('CIECAM02', 'CAM16')
    docstring = docstring.replace('CAM02', 'CAM16')
    docstring = docstring.replace('Luo2006b', 'Li2017')

    docstring = re.match('(.*)Examples', docstring, re.DOTALL).group(1)
    docstring += (
        'Notes\n'
        '    -----\n'
        '    -  This docstring is automatically generated, please refer to\n'
        '       :func:`colour.{0}` definition\n'
        '       for an usage example.'.format(callable_.__name__))

    return docstring


JMh_CAM16_to_UCS_Li2017 = JMh_CIECAM02_to_UCS_Luo2006
JMh_CAM16_to_UCS_Li2017.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(JMh_CIECAM02_to_UCS_Luo2006))

UCS_Li2017_to_JMh_CAM16 = UCS_Luo2006_to_JMh_CIECAM02
UCS_Li2017_to_JMh_CAM16.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(UCS_Luo2006_to_JMh_CIECAM02))

JMh_CAM16_to_CAM16LCD = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
JMh_CAM16_to_CAM16LCD.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(JMh_CIECAM02_to_CAM02LCD))

CAM16LCD_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
CAM16LCD_to_JMh_CAM16.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(CAM02LCD_to_JMh_CIECAM02))

JMh_CAM16_to_CAM16SCD = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])
JMh_CAM16_to_CAM16SCD.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(JMh_CIECAM02_to_CAM02SCD))

CAM16SCD_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])
CAM16SCD_to_JMh_CAM16.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(CAM02SCD_to_JMh_CIECAM02))

JMh_CAM16_to_CAM16UCS = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
JMh_CAM16_to_CAM16UCS.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(JMh_CIECAM02_to_CAM02UCS))

CAM16UCS_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
CAM16UCS_to_JMh_CAM16.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(CAM02UCS_to_JMh_CIECAM02))
