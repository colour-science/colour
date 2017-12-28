#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ocean Optics
=================

Defines input object for *Ocean Optics* spectral data files:

-   :func:`read_spds_from_oceanoptics_file`
"""


from colour.colorimetry import SpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['read_spds_from_oceanoptics_file']


def read_spds_from_oceanoptics_file(file_path):
    """
    Reads the spectral data from given *Ocean Optics* file and returns it as an
    *OrderedDict* of
    :class:`colour.colorimetry.spectrum.SpectralPowerDistribution` classes.

    Parameters
    ----------
    path : unicode
        *OceanOptics* file path.

    Returns
    -------
    OrderedDict
        :class:`colour.colorimetry.spectrum.SpectralPowerDistribution`
        classes of given *X-Rite* file.

    Notes
    -----
    -   This parser assumes the spectrometer is used in Transmission mode,
        and thus values nominally range from 0-100.  Note that due to noise,
        values may appear far outside this range.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> ocean_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                           'resources',
    ...                           'ocean_optics_flame_spectrometer.txt')
    >>> spds_data = read_spds_from_xrite_file(ocean_file)
    >>> # TODO here...  # doctest: +SKIP
    [178.179, 178.558, 178.936, 179.314, 179.693, ...]  # these are the first few keys
    """
    with open(file_path, 'r') as fid:
        txtlines = fid.readlines()

        idx, ready_length, ready_spectral = None, False, False
        for i, line in enumerate(txtlines):
            if 'Number of Pixels in Spectrum' in line:
                ready_length = int(line.split()[-1]), True
            elif '>>>>>Begin Spectral Data<<<<<' in line:
                idx, ready_spectral = i, True

        if not ready_length or not ready_spectral:
            raise IOError('''File lacks line stating "Number of Pixels in Spectrum" or
                             ">>>>>Begin Spectral Data<<<<<" and appears to be corrupt.''')
        data_lines = txtlines[idx + 1:]
        data = {}
        for idx, line in enumerate(data_lines):
            wvl, val = line.split()
            data[float(wvl)] = float(val)

        return SpectralPowerDistribution(name='Sample', data=data)
