#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometry
========

Defines objects related to geometrical computations:

-   :func:`normalise_vector`
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['normalise_vector']


def normalise_vector(v):
    """
    Normalises given vector :math:`v`.

    Parameters
    ----------
    v : array_like
        Vector :math:`v` to normalise.

    Returns
    -------
    ndarray
        Normalised vector :math:`v`.

    Examples
    --------
    >>> v = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> normalise_vector(v)  # doctest: +ELLIPSIS
    array([ 0.4525410...,  0.6470802...,  0.6135908...])
    """

    return v / np.linalg.norm(v)
