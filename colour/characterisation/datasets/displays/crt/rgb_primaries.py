# -*- coding: utf-8 -*-
"""
CRT Displays RGB Primaries
==========================

Defines *CRT* displays *RGB* primaries multi-spectral distributions.

Each *CRT* display data is in the form of a *dict* of
:class:`colour.characterisation.RGB_DisplayPrimaries` classes as follows::

    {'name': RGB_DisplayPrimaries,
    ...,
    'name': RGB_DisplayPrimaries}

The following *CRT* displays are available:

-   Typical CRT Brainard 1997

See Also
--------
`Displays Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/displays.ipynb>`_

References
----------
-   :cite:`Machado2010a` : Machado, G. M. (2010). A model for simulation of
    color vision deficiency and a color contrast enhancement technique for
    dichromats. Retrieved from http://www.lume.ufrgs.br/handle/10183/26950
"""

from __future__ import division, unicode_literals

from colour.characterisation import RGB_DisplayPrimaries
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['CRT_DISPLAYS_RGB_PRIMARIES_DATA', 'CRT_DISPLAYS_RGB_PRIMARIES']

CRT_DISPLAYS_RGB_PRIMARIES_DATA = {
    'Typical CRT Brainard 1997': {
        380.0: (0.0025, 0.0018, 0.0219),
        385.0: (0.0017, 0.0016, 0.0336),
        390.0: (0.0017, 0.0020, 0.0524),
        395.0: (0.0011, 0.0021, 0.0785),
        400.0: (0.0017, 0.0025, 0.1130),
        405.0: (0.0028, 0.0030, 0.1624),
        410.0: (0.0037, 0.0043, 0.2312),
        415.0: (0.0046, 0.0059, 0.3214),
        420.0: (0.0064, 0.0079, 0.4263),
        425.0: (0.0079, 0.0104, 0.5365),
        430.0: (0.0094, 0.0126, 0.6296),
        435.0: (0.0105, 0.0147, 0.6994),
        440.0: (0.0113, 0.0170, 0.7470),
        445.0: (0.0115, 0.0191, 0.7654),
        450.0: (0.0113, 0.0220, 0.7519),
        455.0: (0.0113, 0.0267, 0.7151),
        460.0: (0.0115, 0.0340, 0.6619),
        465.0: (0.0164, 0.0462, 0.5955),
        470.0: (0.0162, 0.0649, 0.5177),
        475.0: (0.0120, 0.0936, 0.4327),
        480.0: (0.0091, 0.1345, 0.3507),
        485.0: (0.0119, 0.1862, 0.2849),
        490.0: (0.0174, 0.2485, 0.2278),
        495.0: (0.0218, 0.3190, 0.1809),
        500.0: (0.0130, 0.3964, 0.1408),
        505.0: (0.0123, 0.4691, 0.1084),
        510.0: (0.0260, 0.5305, 0.0855),
        515.0: (0.0242, 0.5826, 0.0676),
        520.0: (0.0125, 0.6195, 0.0537),
        525.0: (0.0119, 0.6386, 0.0422),
        530.0: (0.0201, 0.6414, 0.0341),
        535.0: (0.0596, 0.6348, 0.0284),
        540.0: (0.0647, 0.6189, 0.0238),
        545.0: (0.0251, 0.5932, 0.0197),
        550.0: (0.0248, 0.5562, 0.0165),
        555.0: (0.0325, 0.5143, 0.0143),
        560.0: (0.0199, 0.4606, 0.0119),
        565.0: (0.0161, 0.3993, 0.0099),
        570.0: (0.0128, 0.3297, 0.0079),
        575.0: (0.0217, 0.2719, 0.0065),
        580.0: (0.0693, 0.2214, 0.0057),
        585.0: (0.1220, 0.1769, 0.0051),
        590.0: (0.1861, 0.1407, 0.0047),
        595.0: (0.2173, 0.1155, 0.0043),
        600.0: (0.0777, 0.0938, 0.0029),
        605.0: (0.0531, 0.0759, 0.0023),
        610.0: (0.2434, 0.0614, 0.0036),
        615.0: (0.5812, 0.0522, 0.0061),
        620.0: (0.9354, 0.0455, 0.0088),
        625.0: (1.6054, 0.0437, 0.0141),
        630.0: (0.6464, 0.0278, 0.0060),
        635.0: (0.1100, 0.0180, 0.0015),
        640.0: (0.0322, 0.0136, 0.0008),
        645.0: (0.0207, 0.0107, 0.0006),
        650.0: (0.0194, 0.0085, 0.0006),
        655.0: (0.0196, 0.0067, 0.0007),
        660.0: (0.0166, 0.0055, 0.0006),
        665.0: (0.0173, 0.0044, 0.0005),
        670.0: (0.0220, 0.0039, 0.0006),
        675.0: (0.0186, 0.0033, 0.0005),
        680.0: (0.0377, 0.0030, 0.0007),
        685.0: (0.0782, 0.0028, 0.0010),
        690.0: (0.0642, 0.0023, 0.0010),
        695.0: (0.1214, 0.0028, 0.0016),
        700.0: (0.7169, 0.0078, 0.0060),
        705.0: (1.1098, 0.0113, 0.0094),
        710.0: (0.3106, 0.0039, 0.0030),
        715.0: (0.0241, 0.0011, 0.0007),
        720.0: (0.0180, 0.0009, 0.0009),
        725.0: (0.0149, 0.0008, 0.0008),
        730.0: (0.0108, 0.0009, 0.0011),
        735.0: (0.0097, 0.0011, 0.0010),
        740.0: (0.0091, 0.0009, 0.0010),
        745.0: (0.0093, 0.0010, 0.0012),
        750.0: (0.0083, 0.0011, 0.0013),
        755.0: (0.0073, 0.0013, 0.0012),
        760.0: (0.0081, 0.0015, 0.0016),
        765.0: (0.0067, 0.0018, 0.0015),
        770.0: (0.0070, 0.0021, 0.0028),
        775.0: (0.0073, 0.0015, 0.0046),
        780.0: (0.0066, 0.0018, 0.0058)
    }
}

CRT_DISPLAYS_RGB_PRIMARIES = CaseInsensitiveMapping({
    'Typical CRT Brainard 1997':
        RGB_DisplayPrimaries(
            CRT_DISPLAYS_RGB_PRIMARIES_DATA['Typical CRT Brainard 1997'],
            name='Typical CRT Brainard 1997')
})
"""
*CRT* displays *RGB* primaries multi-spectral distributions.

References
----------
:cite:`Machado2010a`

CRT_DISPLAYS_RGB_PRIMARIES : CaseInsensitiveMapping
    **{'Typical CRT Brainard 1997'}**
"""
