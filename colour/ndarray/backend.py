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
    'get_ndimensional_array_backend', '_set_ndimensional_array_backend',
    'ndimensional_array_backend', 'NDimensionalArrayBackend'
]

_NDIMENSIONAL_ARRAY_BACKEND = os.environ.get(
    'COLOUR_SCIENCE__NDIMENSIONAL_ARRAY_BACKEND', 'Numpy').lower()


def get_ndimensional_array_backend():

    return _NDIMENSIONAL_ARRAY_BACKEND


def _set_ndimensional_array_backend(backend='Numpy'):
    global _NDIMENSIONAL_ARRAY_BACKEND

    backend = str(backend).lower()
    valid = ('numpy', 'cupy')
    assert backend in valid, 'Scale must be one of "{0}".'.format(valid)

    _NDIMENSIONAL_ARRAY_BACKEND = backend


class ndimensional_array_backend(object):
    def __init__(self, backend):
        self._backend = backend
        self._previous_backend = get_ndimensional_array_backend()

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
            numpyList = dir(numpy)
            cupyList = dir(cupy)
            for i in numpyList:
                if i not in cupyList:
                    self._cupy_unsupported.append(i)
            self._cupy = cupy
        except ImportError:
            pass

    def __getattr__(self, attribute):

        try:
            failsafe = getattr(self._failsafe, attribute)
        except Exception:
            failsafe = None

        if _NDIMENSIONAL_ARRAY_BACKEND == 'numpy':
            if attribute in ['array', 'asarray']:

                def checkForCupy(*args, **kwargs):
                    args = list(args)
                    for i in range(len(args)):
                        if hasattr(self._cupy, 'ndarray') and \
                           isinstance(args[i], self._cupy.ndarray):
                            args[i] = self._cupy.asnumpy(args[i])
                    args = tuple(args)

                    return failsafe(*args, **kwargs)

                return checkForCupy
            return getattr(self._numpy, attribute)
        elif _NDIMENSIONAL_ARRAY_BACKEND == 'cupy' and self._cupy is not None:
            if attribute not in self._cupy_unsupported:
                try:
                    return getattr(self._cupy, attribute)
                except AttributeError:
                    return failsafe
            else:

                def middleware(*args, **kwargs):
                    args = list(args)
                    for i in range(len(args)):
                        if isinstance(args[i], self._cupy.ndarray):
                            args[i] = self._cupy.asnumpy(args[i])
                        else:
                            if isinstance(args[i], tuple):
                                args[i] = list(args[i])
                            try:
                                for z in range(len(args[i])):
                                    if isinstance(args[i][z],
                                                  self._cupy.ndarray):
                                        arr = self._cupy.asnumpy(args[i][z])
                                        args[i][z] = arr
                                args[i] = tuple(args[i])
                            except Exception:
                                pass

                    args = tuple(args)
                    r = failsafe(*args, **kwargs)
                    if isinstance(r, self._numpy.ndarray):
                        return self._cupy.array(r)
                    elif isinstance(r, (list)):
                        for z in range(len(r)):
                            if isinstance(r, self._numpy.ndarray):
                                r[z] = self._cupy.array(r[z])
                    return r

                if callable(failsafe):
                    return middleware
                else:
                    return failsafe
        else:
            return failsafe

    def set_ndimensional_array_backend(self, backend):
        _set_ndimensional_array_backend(backend)
