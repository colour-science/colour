#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chromatic Adaptation Transforms
===============================

Defines various chromatic adaptation transforms (CAT) and objects to
calculate the chromatic adaptation matrix between two given *CIE XYZ*
colourspace matrices:

-   :attr:`XYZ_SCALING_CAT`: *XYZ Scaling* CAT [1]_
-   :attr:`BRADFORD_CAT`: *Bradford* CAT [1]_
-   :attr:`VON_KRIES_CAT`: *Von Kries* CAT [1]_
-   :attr:`FAIRCHILD_CAT`: *Fairchild* CAT [2]_
-   :attr:`CAT02_CAT`: *CAT02* CAT [3]_

See Also
--------
`Chromatic Adaptation Transforms IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/cat.ipynb>`_  # noqa

References
----------
.. [1]  http://brucelindbloom.com/Eqn_ChromAdapt.html
.. [2]  http://rit-mcsl.org/fairchild//files/FairchildYSh.zip
.. [3]  http://en.wikipedia.org/wiki/CIECAM02#CAT02
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_SCALING_CAT',
           'BRADFORD_CAT',
           'VON_KRIES_CAT',
           'FAIRCHILD_CAT',
           'CAT02_CAT',
           'CAT02_INVERSE_CAT',
           'CHROMATIC_ADAPTATION_METHODS',
           'chromatic_adaptation_matrix']

XYZ_SCALING_CAT = np.array(np.identity(3)).reshape((3, 3))
"""
*XYZ Scaling* chromatic adaptation transform. [1]_

XYZ_SCALING_CAT : array_like, (3, 3)
"""

BRADFORD_CAT = np.array(
    [[0.8951000, 0.2664000, -0.1614000],
     [-0.7502000, 1.7135000, 0.0367000],
     [0.0389000, -0.0685000, 1.0296000]])
"""
*Bradford* chromatic adaptation transform. [1]_

BRADFORD_CAT : array_like, (3, 3)
"""

VON_KRIES_CAT = np.array(
    [[0.4002400, 0.7076000, -0.0808100],
     [-0.2263000, 1.1653200, 0.0457000],
     [0.0000000, 0.0000000, 0.9182200]])
"""
*Von Kries* chromatic adaptation transform. [1]_

VON_KRIES_CAT : array_like, (3, 3)
"""

FAIRCHILD_CAT = np.array(
    [[.8562, .3372, -.1934],
     [-.8360, 1.8327, .0033],
     [.0357, -.0469, 1.0112]])
"""
*Fairchild* chromatic adaptation transform. [2]_

FAIRCHILD_CAT : array_like, (3, 3)
"""

CAT02_CAT = np.array(
    [[0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975, 0.0061],
     [0.0030, 0.0136, 0.9834]])
"""
*CAT02* chromatic adaptation transform. [3]_

CAT02_CAT : array_like, (3, 3)
"""

CAT02_INVERSE_CAT = np.linalg.inv(CAT02_CAT)
"""
Inverse *CAT02* chromatic adaptation transform. [3]_

CAT02_INVERSE_CAT : array_like, (3, 3)
"""

CHROMATIC_ADAPTATION_METHODS = CaseInsensitiveMapping(
    {'XYZ Scaling': XYZ_SCALING_CAT,
     'Bradford': BRADFORD_CAT,
     'Von Kries': VON_KRIES_CAT,
     'Fairchild': FAIRCHILD_CAT,
     'CAT02': CAT02_CAT})
"""
Supported chromatic adaptation transform methods.

CHROMATIC_ADAPTATION_METHODS : dict
    ('XYZ Scaling', 'Bradford', 'Von Kries', 'Fairchild, 'CAT02')
"""


def chromatic_adaptation_matrix(XYZ1, XYZ2, method='CAT02'):
    """
    Returns the *chromatic adaptation* matrix from given source and target
    *CIE XYZ* colourspace *array_like* variables.

    Parameters
    ----------
    XYZ1 : array_like, (3,)
        *CIE XYZ* source *array_like* variable.
    XYZ2 : array_like, (3,)
        *CIE XYZ* target *array_like* variable.
    method : unicode, optional
        ('XYZ Scaling', 'Bradford', 'Von Kries', 'Fairchild, 'CAT02'),
        Chromatic adaptation method.

    Returns
    -------
    ndarray, (3, 3)
        Chromatic adaptation matrix.

    Raises
    ------
    KeyError
        If chromatic adaptation method is not defined.

    References
    ----------
    .. [4]  http://brucelindbloom.com/Eqn_ChromAdapt.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> XYZ1 = np.array([1.09923822, 1.000, 0.35445412])
    >>> XYZ2 = np.array([0.96907232, 1.000, 1.121792157])
    >>> chromatic_adaptation_matrix(XYZ1, XYZ2)  # doctest: +ELLIPSIS
    array([[ 0.8714561..., -0.1320467...,  0.4039483...],
           [-0.0963880...,  1.0490978...,  0.160403... ],
           [ 0.0080207...,  0.0282636...,  3.0602319...]])

    Using *Bradford* method:

    >>> XYZ1 = np.array([1.09923822, 1.000, 0.35445412])
    >>> XYZ2 = np.array([0.96907232, 1.000, 1.121792157])
    >>> method = 'Bradford'
    >>> chromatic_adaptation_matrix(XYZ1, XYZ2, method)  # doctest: +ELLIPSIS
    array([[ 0.8518131..., -0.1134786...,  0.4124804...],
           [-0.1277659...,  1.0928930...,  0.1341559...],
           [ 0.0845323..., -0.1434969...,  3.3075309...]])
    """

    method_matrix = CHROMATIC_ADAPTATION_METHODS.get(method)

    if method_matrix is None:
        raise KeyError(
            '"{0}" chromatic adaptation method is not defined! Supported '
            'methods: "{1}".'.format(method,
                                     CHROMATIC_ADAPTATION_METHODS.keys()))

    XYZ1, XYZ2 = np.ravel(XYZ1), np.ravel(XYZ2)

    if (XYZ1 == XYZ2).all():
        # Skip the chromatic adaptation computation if the two input matrices
        # are the same, because no adaptation is needed.
        return np.identity(3)

    rgb_source = np.ravel(np.dot(method_matrix, XYZ1))
    rgb_target = np.ravel(np.dot(method_matrix, XYZ2))
    crd = np.diagflat(np.array(
        [[rgb_target[0] / rgb_source[0],
          rgb_target[1] / rgb_source[1],
          rgb_target[2] / rgb_source[2]]])).reshape((3, 3))
    cat = np.dot(np.dot(np.linalg.inv(method_matrix), crd), method_matrix)

    return cat
