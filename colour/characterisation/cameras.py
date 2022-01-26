# -*- coding: utf-8 -*-
"""
Cameras Sensitivities
=====================

Defines the spectral distributions classes for the datasets from
the :mod:`colour.characterisation.datasets.cameras` module:

-   :class:`colour.characterisation.RGB_CameraSensitivities`: Implements
    support for a camera *RGB* sensitivities.
"""

from __future__ import annotations

from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
)
from colour.continuous import MultiSignals, Signal
from colour.hints import ArrayLike, Any, Optional, Sequence, Union
from colour.utilities import is_pandas_installed

if is_pandas_installed():
    from pandas import DataFrame, Series
else:  # pragma: no cover
    from unittest import mock

    DataFrame = mock.MagicMock()
    Series = mock.MagicMock()

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RGB_CameraSensitivities',
]


class RGB_CameraSensitivities(MultiSpectralDistributions):
    """
    Implements support for a camera *RGB* sensitivities.

    Parameters
    ----------
    data
        Data to be stored in the multi-spectral distributions.
    domain
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        :attr:`colour.continuous.Signal.wavelengths` attribute.
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    name
       Multi-spectral distributions name.
    interpolator
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.SpectralDistribution` class instances.
    extrapolator
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.SpectralDistribution` class instances.
    strict_labels
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.colorimetry.LMS_ConeFundamentals.labels` attribute value.
    """

    def __init__(
            self,
            data: Optional[Union[ArrayLike, DataFrame, dict, MultiSignals,
                                 MultiSpectralDistributions, Sequence, Series,
                                 Signal, SpectralDistribution]] = None,
            domain: Optional[Union[ArrayLike, SpectralShape]] = None,
            labels: Optional[Sequence] = None,
            **kwargs: Any):
        super(RGB_CameraSensitivities, self).__init__(
            data, domain, labels=('red', 'green', 'blue'), **kwargs)
