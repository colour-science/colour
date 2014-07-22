#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**runtime.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package runtime cache through the :class:`RuntimeCache` class.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RuntimeCache"]


class RuntimeCache():
    """
    Defines **Colour** package runtime cache.
    """

    munsell_specifications = None
    """Munsell specifications."""

    wavelength_to_XYZ = {}
    """Wavelength to *CIE XYZ* colourspace matrices."""

    planck_law = {}
    """Blackbody spectral radiance."""

    XYZ_optimal_colour_stimuli = {}
    """Optima colour stimuli in *CIE XYZ* colourspace."""

    XYZ_optimal_colour_stimuli_triangulations = {}
    """Optima colour stimuli triangulations in *CIE XYZ* colourspace."""