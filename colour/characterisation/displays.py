"""
RGB Display Primaries
=====================

Defines the spectral distributions classes for the datasets from
the :mod:`colour.characterisation.datasets.displays` module:

-   :class:`colour.characterisation.RGB_DisplayPrimaries`: Implements support
    for a *RGB* display (such as a *CRT* or *LCD*) primaries multi-spectral
    distributions.
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

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_DisplayPrimaries",
]


class RGB_DisplayPrimaries(MultiSpectralDistributions):
    """
    Implement support for a *RGB* display (such as a *CRT* or *LCD*)
    primaries multi-spectral distributions.

    Parameters
    ----------
    data
        Data to be stored in the multi-spectral distributions.
    domain
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        :attr:`colour.continuous.Signal.wavelengths` property.
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    extrapolator
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.SpectralDistribution` class instances.
    interpolator
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.SpectralDistribution` class instances.
    name
       Multi-spectral distributions name.
    strict_labels
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.colorimetry.RGB_DisplayPrimaries.labels` property
        value.
    """

    def __init__(
        self,
        data: Optional[
            Union[
                ArrayLike,
                DataFrame,
                dict,
                MultiSignals,
                MultiSpectralDistributions,
                Sequence,
                Series,
                Signal,
                SpectralDistribution,
            ]
        ] = None,
        domain: Optional[Union[ArrayLike, SpectralShape]] = None,
        labels: Optional[Sequence] = None,
        **kwargs: Any,
    ):
        super().__init__(
            data, domain, labels=("red", "green", "blue"), **kwargs
        )
