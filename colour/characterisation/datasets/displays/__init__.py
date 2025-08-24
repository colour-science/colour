"""
References
----------
-   :cite:`Fairchild1998b` : Fairchild, M., & Wyble, D. (1998). Colorimetric
    Characterization of The Apple Studio Display (flat panel LCD) (p. 22).
    http://scholarworks.rit.edu/cgi/viewcontent.cgi?article=1922&\
context=article
-   :cite:`Machado2010a` : Machado, Gustavo Mello. (2010). A model for
    simulation of color vision deficiency and a color contrast enhancement
    technique for dichromats. (pp. 1-94).
    http://www.lume.ufrgs.br/handle/10183/26950
"""

from colour.utilities import LazyCanonicalMapping

from .crt import MSDS_DISPLAY_PRIMARIES_CRT
from .lcd import MSDS_DISPLAY_PRIMARIES_LCD

MSDS_DISPLAY_PRIMARIES = LazyCanonicalMapping(MSDS_DISPLAY_PRIMARIES_CRT)
MSDS_DISPLAY_PRIMARIES.update(MSDS_DISPLAY_PRIMARIES_LCD)
MSDS_DISPLAY_PRIMARIES.__doc__ = """
Multi-spectral distributions of display primaries.

References
----------
:cite:`Fairchild1998b`, :cite:`Machado2010a`
"""

__all__ = [
    "MSDS_DISPLAY_PRIMARIES_CRT",
]
__all__ += [
    "MSDS_DISPLAY_PRIMARIES_LCD",
]
