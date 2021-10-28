# -*- coding: utf-8 -*-
"""
LUT Operator
============

Defines the *LUT* operator classes:

-   :class:`colour.io.AbstractLUTSequenceOperator`
"""

from abc import ABC, abstractmethod

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['AbstractLUTSequenceOperator']


class AbstractLUTSequenceOperator(ABC):
    """
    Defines the base class for *LUT* sequence operators.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    Methods
    -------
    -   :meth:`~colour.io.luts.lut.AbstractLUTSequenceOperator.apply`
    """

    @abstractmethod
    def apply(self, RGB, *args):
        """
        Applies the *LUT* sequence operator to given *RGB* colourspace array.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* sequence operator onto.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.
        """

        pass
