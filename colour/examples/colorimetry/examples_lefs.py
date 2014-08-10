#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *luminous efficiency functions* computations.
"""

import pprint
import colour

# Displaying available *luminous efficiency functions*.
pprint.pprint(colour.LEFS)

# Calculating a mesopic luminous efficiency functions.
print(colour.mesopic_luminous_efficiency_function(0.2).values)
