# -*- coding: utf-8 -*-
"""
N-Dimensional Array Computations Backend
========================================

Defines the objects enabling the various n-dimensional array computations
backend.
"""

from __future__ import division, unicode_literals

import inspect
import os
import functools

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'get_ndimensional_array_backend', 'set_ndimensional_array_backend',
    'ndimensional_array_backend', 'NDimensionalArrayBackend'
]

_NDIMENSIONAL_ARRAY_BACKEND = os.environ.get(
    'COLOUR_SCIENCE__NDIMENSIONAL_ARRAY_BACKEND', 'Numpy').lower()


def get_ndimensional_array_backend():

    return _NDIMENSIONAL_ARRAY_BACKEND


def set_ndimensional_array_backend(backend='Numpy'):
    global _NDIMENSIONAL_ARRAY_BACKEND

    backend = str(backend).lower()
    valid = ('numpy', 'jax')
    assert backend in valid, 'Scale must be one of "{0}".'.format(valid)

    _NDIMENSIONAL_ARRAY_BACKEND = backend


class ndimensional_array_backend(object):
    def __init__(self, backend):
        self._backend = backend
        self._previous_backend = get_ndimensional_array_backend()

    def __enter__(self):
        set_ndimensional_array_backend(self._backend)

        return self

    def __exit__(self, *args):
        set_ndimensional_array_backend(self._previous_backend)

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

        # Jax
        self._jax = None
        self._jax_unsupported = []
        try:
            import jax.numpy

            self._jax = jax.numpy
            for name in dir(jax.numpy):
                function = getattr(jax.numpy, name)
                try:
                    source = inspect.getsource(function)
                except (TypeError, OSError):
                    continue

                if "Numpy function {} not yet implemented" in source:
                    self._jax_unsupported.append(name)

            # TypeError: pad() got an unexpected keyword argument 'end_values'
            self._jax_unsupported.append('pad')
            # TypeError: full() takes from 2 to 3 positional arguments
            # but 4 were given
            self._jax_unsupported.append('full')
        except ImportError:
            pass

    def __getattr__(self, attribute):
        failsafe = getattr(self._failsafe, attribute)

        if _NDIMENSIONAL_ARRAY_BACKEND == 'numpy':
            return getattr(self._numpy, attribute)
        elif _NDIMENSIONAL_ARRAY_BACKEND == 'jax' and self._jax is not None:
            if attribute not in self._jax_unsupported:
                try:
                    return getattr(self._jax, attribute)
                except AttributeError:
                    return failsafe
            else:
                return failsafe
        else:
            return failsafe
