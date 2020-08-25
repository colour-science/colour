# -*- coding: utf-8 -*-
"""
Sony .spimtx LUT Format Input / Output Utilities
================================================

Defines *Sony* *.spimtx* *LUT* Format related input / output utilities objects.

-   :func:`colour.io.read_LUT_SonySPImtx`
-   :func:`colour.io.write_LUT_SonySPImtx`
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import re

from colour.io.luts import Matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['read_LUT_SonySPImtx', 'write_LUT_SonySPImtx']


def read_LUT_SonySPImtx(path):
    array = np.loadtxt(path)
    array = array.reshape(3, 4)
    # TODO: Update with "develop" generic function.
    title = re.sub('_|-|\\.', ' ', os.path.splitext(os.path.basename(path))[0])

    return Matrix(array, title)


def write_LUT_SonySPImtx(matrix, path, decimals=6):
    if matrix.array.shape == (3, 4):
        array = matrix.array
    else:
        array = np.hstack([matrix.array, np.zeros((3, 1))])

    np.savetxt(path, array, fmt='%.{0}f'.format(decimals).encode('utf-8'))
