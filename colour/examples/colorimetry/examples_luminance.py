#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *Luminance* computations.
"""

from __future__ import division, unicode_literals

import colour

# Retrieving *luminance* of given *Munsell* value with
# *Newhall, Nickerson, and Judd* 1943 method.
print(colour.luminance_newhall1943(3.74629715382))

# Retrieving *luminance* of given *Lightness* with *1976* method.
print(colour.luminance_1976(37.9856290977))

# Retrieving *luminance* of given *Munsell* value with
# *ASTM D1535-08e1* 2008 method.
print(colour.luminance_ASTM_D1535_08(3.74629715382))

# Retrieving *luminance* using the wrapper:
print(colour.luminance(37.9856290977))
print(colour.luminance(3.74629715382, method='Luminance ASTM D1535-08'))
