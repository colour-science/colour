# -*- coding: utf-8 -*-
"""
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

from __future__ import absolute_import

from .crt import CRT_DISPLAY_RGB_PRIMARIES
from .lcd import LCD_DISPLAY_RGB_PRIMARIES
from colour.utilities import CaseInsensitiveMapping

DISPLAY_RGB_PRIMARIES = CaseInsensitiveMapping(CRT_DISPLAY_RGB_PRIMARIES)
DISPLAY_RGB_PRIMARIES.update(LCD_DISPLAY_RGB_PRIMARIES)
DISPLAY_RGB_PRIMARIES.__doc__ = """
Display *RGB* primaries multi-spectral distributions.

References
----------
:cite:`Fairchild1998b`, :cite:`Machado2010a`

DISPLAY_RGB_PRIMARIES : CaseInsensitiveMapping
    **{Apple Studio Display, Typical CRT Brainard 1997}**
"""

__all__ = ['DISPLAY_RGB_PRIMARIES']
