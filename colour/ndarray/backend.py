# -*- coding: utf-8 -*-
"""
N-Dimensional Array Computations Backend
========================================

Defines the objects enabling the various n-dimensional array computations
backend.
"""

from __future__ import division, unicode_literals

import os
import functools

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'NDimensionalArrayBackend'
]

_NDIMENSIONAL_ARRAY_BACKEND = os.environ.get(
    'COLOUR_SCIENCE__NDIMENSIONAL_ARRAY_BACKEND', 'Cupy').lower()


def _get_ndimensional_array_backend():

    return _NDIMENSIONAL_ARRAY_BACKEND


def _set_ndimensional_array_backend(backend='Numpy'):
    global _NDIMENSIONAL_ARRAY_BACKEND

    backend = str(backend).lower()
    valid = ('numpy', 'cupy')
    assert backend in valid, 'Scale must be one of "{0}".'.format(valid)

    _NDIMENSIONAL_ARRAY_BACKEND = backend


class _ndarray_backend(object):
    def __init__(self, backend):
        self._backend = backend
        self._previous_backend = _get_ndimensional_array_backend()

    def __enter__(self):
        _set_ndimensional_array_backend(self._backend)

        return self

    def __exit__(self, *args):
        _set_ndimensional_array_backend(self._previous_backend)

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


class NDimensionalArrayBackend(object):
    def __init__(self):
        # Numpy
        import numpy

        self._failsafe = self._numpy = numpy

        # CuPy
        self._cupy = None
        self._cupy_unsupported = []
        try:
            import cupy
            self._cupy = cupy
            for i in dir(numpy):
                if i not in dir(cupy):
                    self._cupy_unsupported.append(i)
        except ImportError:
            pass

    def __getattr__(self, attribute):
        try:
            failsafe = getattr(self._failsafe, attribute)
        except AttributeError:
            failsafe = None

        if _NDIMENSIONAL_ARRAY_BACKEND == 'numpy':
            return getattr(self._numpy, attribute)

        elif _NDIMENSIONAL_ARRAY_BACKEND == 'cupy' and self._cupy is not None:
            if attribute not in self._cupy_unsupported:
                try:
                    return getattr(self._cupy, attribute)
                except AttributeError:
                    return failsafe
            else:
                return failsafe
        
        else:
            return failsafe

    def ndarray_backend(self, backend):
        return _ndarray_backend(backend)

    def get_ndimensional_array_backend(self):
        return _get_ndimensional_array_backend()

    def set_ndimensional_array_backend(self, backend='Numpy'):
        return _set_ndimensional_array_backend(backend)
