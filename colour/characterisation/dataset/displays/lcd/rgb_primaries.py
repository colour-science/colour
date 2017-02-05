#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LCD Displays RGB Primaries
==========================

Defines *LCD* displays *RGB* primaries tri-spectral power distributions.

Each *LCD* display data is in the form of a *dict* of
:class:`colour.characterisation.displays.RGB_DisplayPrimaries` classes as
follows::

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
.. [1]  Machado, G. (2010). A model for simulation of color vision deficiency
        and a color contrast enhancement technique for dichromats. Retrieved
        from http://www.lume.ufrgs.br/handle/10183/26950
.. [2]  Fairchild, M., & Wyble, D. (1998). Colorimetric Characterization of
        The Apple Studio Display (flat panel LCD), 22. Retrieved from
        https://ritdml.rit.edu/handle/1850/4368
"""

from __future__ import division, unicode_literals

from colour.characterisation import RGB_DisplayPrimaries
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LCD_DISPLAYS_RGB_PRIMARIES_DATA',
           'LCD_DISPLAYS_RGB_PRIMARIES']

LCD_DISPLAYS_RGB_PRIMARIES_DATA = {
    'Apple Studio Display': {
        'red': {
            380: 0.0000,
            385: 0.0000,
            390: 0.0000,
            395: 0.0000,
            400: 0.0000,
            405: 0.0000,
            410: 0.0000,
            415: 0.0000,
            420: 0.0000,
            425: 0.0000,
            430: 0.0000,
            435: 0.0000,
            440: 0.0040,
            445: 0.0040,
            450: 0.0000,
            455: 0.0040,
            460: 0.0000,
            465: 0.0040,
            470: 0.0040,
            475: 0.0040,
            480: 0.0040,
            485: 0.0040,
            490: 0.0040,
            495: 0.0040,
            500: 0.0040,
            505: 0.0000,
            510: 0.0000,
            515: 0.0000,
            520: 0.0000,
            525: 0.0000,
            530: 0.0000,
            535: 0.0000,
            540: 0.0000,
            545: 0.0437,
            550: 0.0317,
            555: 0.0040,
            560: 0.0000,
            565: 0.0000,
            570: 0.0000,
            575: 0.0000,
            580: 0.0040,
            585: 0.0198,
            590: 0.0635,
            595: 0.0873,
            600: 0.0635,
            605: 0.0714,
            610: 0.2619,
            615: 1.0714,
            620: 0.4881,
            625: 0.3532,
            630: 0.2103,
            635: 0.1944,
            640: 0.0556,
            645: 0.0238,
            650: 0.0476,
            655: 0.0675,
            660: 0.0238,
            665: 0.0397,
            670: 0.0397,
            675: 0.0278,
            680: 0.0278,
            685: 0.0317,
            690: 0.0317,
            695: 0.0198,
            700: 0.0159,
            705: 0.0119,
            710: 0.0952,
            715: 0.0952,
            720: 0.0159,
            725: 0.0040,
            730: 0.0000,
            735: 0.0000,
            740: 0.0000,
            745: 0.0000,
            750: 0.0000,
            755: 0.0000,
            760: 0.0000,
            765: 0.0000,
            770: 0.0000,
            775: 0.0000,
            780: 0.0000},
        'green': {
            380: 0.0000,
            385: 0.0000,
            390: 0.0000,
            395: 0.0000,
            400: 0.0000,
            405: 0.0000,
            410: 0.0000,
            415: 0.0000,
            420: 0.0000,
            425: 0.0000,
            430: 0.0040,
            435: 0.0119,
            440: 0.0000,
            445: 0.0119,
            450: 0.0119,
            455: 0.0079,
            460: 0.0198,
            465: 0.0238,
            470: 0.0317,
            475: 0.0357,
            480: 0.0516,
            485: 0.0873,
            490: 0.0873,
            495: 0.0675,
            500: 0.0437,
            505: 0.0357,
            510: 0.0317,
            515: 0.0317,
            520: 0.0238,
            525: 0.0238,
            530: 0.0317,
            535: 0.1944,
            540: 1.5794,
            545: 1.4048,
            550: 0.4127,
            555: 0.0952,
            560: 0.0317,
            565: 0.0159,
            570: 0.0079,
            575: 0.0952,
            580: 0.1429,
            585: 0.1468,
            590: 0.0754,
            595: 0.0357,
            600: 0.0159,
            605: 0.0040,
            610: 0.0476,
            615: 0.0159,
            620: 0.0040,
            625: 0.0040,
            630: 0.0000,
            635: 0.0000,
            640: 0.0000,
            645: 0.0000,
            650: 0.0000,
            655: 0.0000,
            660: 0.0000,
            665: 0.0040,
            670: 0.0040,
            675: 0.0000,
            680: 0.0000,
            685: 0.0000,
            690: 0.0000,
            695: 0.0000,
            700: 0.0000,
            705: 0.0000,
            710: 0.0040,
            715: 0.0000,
            720: 0.0000,
            725: 0.0000,
            730: 0.0000,
            735: 0.0000,
            740: 0.0000,
            745: 0.0000,
            750: 0.0000,
            755: 0.0000,
            760: 0.0000,
            765: 0.0000,
            770: 0.0000,
            775: 0.0119,
            780: 0.0000},
        'blue': {
            380: 0.0000,
            385: 0.0000,
            390: 0.0000,
            395: 0.0000,
            400: 0.0040,
            405: 0.0040,
            410: 0.0079,
            415: 0.0238,
            420: 0.0516,
            425: 0.0992,
            430: 0.1865,
            435: 0.3929,
            440: 0.2540,
            445: 0.2738,
            450: 0.3016,
            455: 0.3016,
            460: 0.2976,
            465: 0.2698,
            470: 0.2460,
            475: 0.2103,
            480: 0.2460,
            485: 0.3929,
            490: 0.3333,
            495: 0.2024,
            500: 0.0913,
            505: 0.0437,
            510: 0.0238,
            515: 0.0119,
            520: 0.0079,
            525: 0.0040,
            530: 0.0040,
            535: 0.0159,
            540: 0.0794,
            545: 0.0754,
            550: 0.0079,
            555: 0.0040,
            560: 0.0000,
            565: 0.0000,
            570: 0.0000,
            575: 0.0000,
            580: 0.0000,
            585: 0.0000,
            590: 0.0000,
            595: 0.0000,
            600: 0.0000,
            605: 0.0000,
            610: 0.0000,
            615: 0.0000,
            620: 0.0000,
            625: 0.0000,
            630: 0.0000,
            635: 0.0000,
            640: 0.0000,
            645: 0.0000,
            650: 0.0000,
            655: 0.0000,
            660: 0.0000,
            665: 0.0000,
            670: 0.0000,
            675: 0.0000,
            680: 0.0000,
            685: 0.0000,
            690: 0.0000,
            695: 0.0000,
            700: 0.0000,
            705: 0.0000,
            710: 0.0000,
            715: 0.0000,
            720: 0.0000,
            725: 0.0000,
            730: 0.0000,
            735: 0.0000,
            740: 0.0000,
            745: 0.0000,
            750: 0.0000,
            755: 0.0000,
            760: 0.0000,
            765: 0.0000,
            770: 0.0000,
            775: 0.0000,
            780: 0.0000}}}

LCD_DISPLAYS_RGB_PRIMARIES = CaseInsensitiveMapping(
    {'Apple Studio Display': RGB_DisplayPrimaries(
        'Apple Studio Display',
        LCD_DISPLAYS_RGB_PRIMARIES_DATA['Apple Studio Display'])})
"""
*LCD* displays *RGB* primaries tri-spectral power distributions.

LCD_DISPLAYS_RGB_PRIMARIES : CaseInsensitiveMapping
    **{'Apple Studio Display'}**
"""
