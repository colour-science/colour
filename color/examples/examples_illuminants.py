#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shows some **Color** package *illuminants* related examples.
"""

import color

# Retrieving *D60* illuminant chromaticity coordinates.
print(color.ILLUMINANTS["Standard CIE 1931 2 Degree Observer"]["D60"])

# Displaying all illuminants data per standard observers.
for observer, illuminants in color.ILLUMINANTS.iteritems():
    print("Observer: '{0}'.".format(observer))
    for illuminant, xy in illuminants.iteritems():
        print("'{0}': {1}".format(illuminant, xy))
