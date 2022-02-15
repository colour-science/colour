"""
X-Rite Data Input
=================

Defines the input object for *X-Rite* spectral data files:

-   :func:`colour.read_sds_from_xrite_file`
"""

from __future__ import annotations

import codecs
import re

from colour.colorimetry import SpectralDistribution
from colour.hints import Dict

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XRITE_FILE_ENCODING",
    "read_sds_from_xrite_file",
]

XRITE_FILE_ENCODING: str = "utf-8"


def read_sds_from_xrite_file(path: str) -> Dict[str, SpectralDistribution]:
    """
    Read the spectral data from given *X-Rite* file and returns it as a
    *dict* of :class:`colour.SpectralDistribution` class instances.

    Parameters
    ----------
    path
        Absolute *X-Rite* file path.

    Returns
    -------
    :class:`dict`
        *Dict* of :class:`colour.SpectralDistribution` class instances.

    Notes
    -----
    -   This parser is minimalistic and absolutely not bullet-proof.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> xrite_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                           'resources',
    ...                           'X-Rite_Digital_Colour_Checker.txt')
    >>> sds_data = read_sds_from_xrite_file(xrite_file)
    >>> pprint(list(sds_data.keys()))  # doctest: +SKIP
    ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    """

    with codecs.open(path, encoding=XRITE_FILE_ENCODING) as xrite_file:
        lines = xrite_file.read().strip().split("\n")

        xrite_sds = {}
        is_spectral_data_format, is_spectral_data = False, False
        for line in lines:
            line = line.strip()

            if line == "END_DATA_FORMAT":
                is_spectral_data_format = False

            if line == "END_DATA":
                is_spectral_data = False

            if is_spectral_data_format:
                wavelengths = [x for x in re.findall("nm(\\d+)", line)]
                index = len(wavelengths)

            if is_spectral_data:
                tokens = line.split()
                xrite_sds[tokens[1]] = SpectralDistribution(
                    tokens[-index:], wavelengths, name=tokens[1]
                )

            if line == "BEGIN_DATA_FORMAT":
                is_spectral_data_format = True

            if line == "BEGIN_DATA":
                is_spectral_data = True

        return xrite_sds
