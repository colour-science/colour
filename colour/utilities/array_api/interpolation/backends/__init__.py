"""
Backend-specialised interpolator implementations.

Each submodule (``numpy``, ``torch``, ``jax``) provides identically named
interpolator classes (``Linear``, ``NearestNeighbour``, ``Null``, ``Sprague``,
``CubicSpline``, ``Pchip``, ``Kernel`` and the ``Extrapolator``) specialised for
that array backend. They are imported lazily by
:mod:`colour.utilities.array_api.interpolation._dispatch`. The kernel functions
shared by the ``Kernel`` interpolators live in :mod:`._kernels`.
"""

from __future__ import annotations

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"
