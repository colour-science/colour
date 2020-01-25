# -*- coding: utf-8 -*-
"""
LCD Displays RGB Primaries
==========================

Defines *LCD* displays *RGB* primaries multi-spectral distributions.

Each *LCD* display data is in the form of a *dict* of
:class:`colour.characterisation.RGB_DisplayPrimaries` classes as follows::

    {'name': RGB_DisplayPrimaries,
    ...,
    'name': RGB_DisplayPrimaries}

The following *LCD* displays are available:

-   Apple Studio Display

See Also
--------
`Displays Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/displays.ipynb>`_

References
----------
-   :cite:`Fairchild1998b` : Fairchild, M., & Wyble, D. (1998). Colorimetric
    Characterization of The Apple Studio Display (flat panel LCD). Retrieved
    from http://scholarworks.rit.edu/cgi/viewcontent.cgi?article=1922&\
context=article
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

__all__ = ['LCD_DISPLAYS_RGB_PRIMARIES_DATA', 'LCD_DISPLAYS_RGB_PRIMARIES']

LCD_DISPLAYS_RGB_PRIMARIES_DATA = {
    'Apple Studio Display': {
        380: (0.0000, 0.0000, 0.0000),
        385: (0.0000, 0.0000, 0.0000),
        390: (0.0000, 0.0000, 0.0000),
        395: (0.0000, 0.0000, 0.0000),
        400: (0.0000, 0.0000, 0.0040),
        405: (0.0000, 0.0000, 0.0040),
        410: (0.0000, 0.0000, 0.0079),
        415: (0.0000, 0.0000, 0.0238),
        420: (0.0000, 0.0000, 0.0516),
        425: (0.0000, 0.0000, 0.0992),
        430: (0.0000, 0.0040, 0.1865),
        435: (0.0000, 0.0119, 0.3929),
        440: (0.0040, 0.0000, 0.2540),
        445: (0.0040, 0.0119, 0.2738),
        450: (0.0000, 0.0119, 0.3016),
        455: (0.0040, 0.0079, 0.3016),
        460: (0.0000, 0.0198, 0.2976),
        465: (0.0040, 0.0238, 0.2698),
        470: (0.0040, 0.0317, 0.2460),
        475: (0.0040, 0.0357, 0.2103),
        480: (0.0040, 0.0516, 0.2460),
        485: (0.0040, 0.0873, 0.3929),
        490: (0.0040, 0.0873, 0.3333),
        495: (0.0040, 0.0675, 0.2024),
        500: (0.0040, 0.0437, 0.0913),
        505: (0.0000, 0.0357, 0.0437),
        510: (0.0000, 0.0317, 0.0238),
        515: (0.0000, 0.0317, 0.0119),
        520: (0.0000, 0.0238, 0.0079),
        525: (0.0000, 0.0238, 0.0040),
        530: (0.0000, 0.0317, 0.0040),
        535: (0.0000, 0.1944, 0.0159),
        540: (0.0000, 1.5794, 0.0794),
        545: (0.0437, 1.4048, 0.0754),
        550: (0.0317, 0.4127, 0.0079),
        555: (0.0040, 0.0952, 0.0040),
        560: (0.0000, 0.0317, 0.0000),
        565: (0.0000, 0.0159, 0.0000),
        570: (0.0000, 0.0079, 0.0000),
        575: (0.0000, 0.0952, 0.0000),
        580: (0.0040, 0.1429, 0.0000),
        585: (0.0198, 0.1468, 0.0000),
        590: (0.0635, 0.0754, 0.0000),
        595: (0.0873, 0.0357, 0.0000),
        600: (0.0635, 0.0159, 0.0000),
        605: (0.0714, 0.0040, 0.0000),
        610: (0.2619, 0.0476, 0.0000),
        615: (1.0714, 0.0159, 0.0000),
        620: (0.4881, 0.0040, 0.0000),
        625: (0.3532, 0.0040, 0.0000),
        630: (0.2103, 0.0000, 0.0000),
        635: (0.1944, 0.0000, 0.0000),
        640: (0.0556, 0.0000, 0.0000),
        645: (0.0238, 0.0000, 0.0000),
        650: (0.0476, 0.0000, 0.0000),
        655: (0.0675, 0.0000, 0.0000),
        660: (0.0238, 0.0000, 0.0000),
        665: (0.0397, 0.0040, 0.0000),
        670: (0.0397, 0.0040, 0.0000),
        675: (0.0278, 0.0000, 0.0000),
        680: (0.0278, 0.0000, 0.0000),
        685: (0.0317, 0.0000, 0.0000),
        690: (0.0317, 0.0000, 0.0000),
        695: (0.0198, 0.0000, 0.0000),
        700: (0.0159, 0.0000, 0.0000),
        705: (0.0119, 0.0000, 0.0000),
        710: (0.0952, 0.0040, 0.0000),
        715: (0.0952, 0.0000, 0.0000),
        720: (0.0159, 0.0000, 0.0000),
        725: (0.0040, 0.0000, 0.0000),
        730: (0.0000, 0.0000, 0.0000),
        735: (0.0000, 0.0000, 0.0000),
        740: (0.0000, 0.0000, 0.0000),
        745: (0.0000, 0.0000, 0.0000),
        750: (0.0000, 0.0000, 0.0000),
        755: (0.0000, 0.0000, 0.0000),
        760: (0.0000, 0.0000, 0.0000),
        765: (0.0000, 0.0000, 0.0000),
        770: (0.0000, 0.0000, 0.0000),
        775: (0.0000, 0.0119, 0.0000),
        780: (0.0000, 0.0000, 0.0000)
    }
}

LCD_DISPLAYS_RGB_PRIMARIES = CaseInsensitiveMapping({
    'Apple Studio Display':
        RGB_DisplayPrimaries(
            LCD_DISPLAYS_RGB_PRIMARIES_DATA['Apple Studio Display'],
            name='Apple Studio Display')
})
"""
*LCD* displays *RGB* primaries multi-spectral distributions.

References
----------
:cite:`Fairchild1998b`, :cite:`Machado2010a`

LCD_DISPLAYS_RGB_PRIMARIES : CaseInsensitiveMapping
    **{'Apple Studio Display'}**
"""
