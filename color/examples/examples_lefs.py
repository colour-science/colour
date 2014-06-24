# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package *luminous efficiency functions* related examples.
"""

import pprint
import color

# Displaying available *luminous efficiency functions*.
pprint.pprint(color.LEFS)

# Calculating a mesopic luminous efficiency functions.
print(color.mesopic_luminous_efficiency_function(0.2).values)
