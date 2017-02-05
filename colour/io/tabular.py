#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Tabular Data Input / Output
===============================

Defines various input / output objects for *CSV* tabular data files:

-   :func:`read_spectral_data_from_csv_file`
-   :func:`read_spds_from_csv_file`
-   :func:`write_spds_to_csv_file`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import OrderedDict
import csv

from colour.colorimetry import SpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['read_spectral_data_from_csv_file',
           'read_spds_from_csv_file',
           'write_spds_to_csv_file']


def read_spectral_data_from_csv_file(path,
                                     delimiter=',',
                                     fields=None,
                                     default=0):
    """
    Reads the spectral data from given *CSV* file in the following form:

    390,  4.15003E-04,  3.68349E-04,  9.54729E-03
    395,  1.05192E-03,  9.58658E-04,  2.38250E-02
    400,  2.40836E-03,  2.26991E-03,  5.66498E-02
    ...
    830,  9.74306E-07,  9.53411E-08,  0.00000

    and returns it as an *OrderedDict* of *dict* as follows:

    OrderedDict([
    ('field', {'wavelength': 'value', ..., 'wavelength': 'value'}),
    ...,
    ('field', {'wavelength': 'value', ..., 'wavelength': 'value'})])

    Parameters
    ----------
    path : unicode
        Absolute *CSV* file path.
    delimiter : unicode, optional
        *CSV* file content delimiter.
    fields : array_like, optional
        *CSV* file spectral data fields names. If no value is provided the
        first line of the file will be used as spectral data fields names.
    default : numeric, optional
        Default value for fields row with missing value.

    Returns
    -------
    OrderedDict
        *CSV* file content.

    Raises
    ------
    RuntimeError
        If the *CSV* spectral data file doesn't define the appropriate fields.

    Notes
    -----
    -   A *CSV* spectral data file should define at least define two fields:
        one for the wavelengths and one for the associated values of one
        spectral power distribution.
    -   If no value is provided for the fields names, the first line of the
        file will be used as spectral data fields names.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> csv_file = os.path.join(
    ...     os.path.dirname(__file__),
    ...     'tests',
    ...     'resources',
    ...     'colorchecker_n_ohta.csv')
    >>> spds_data = read_spectral_data_from_csv_file(csv_file)
    >>> pprint(list(spds_data.keys()))
    ['1',
     '2',
     '3',
     '4',
     '5',
     '6',
     '7',
     '8',
     '9',
     '10',
     '11',
     '12',
     '13',
     '14',
     '15',
     '16',
     '17',
     '18',
     '19',
     '20',
     '21',
     '22',
     '23',
     '24']
    """

    with open(path, 'rU') as csv_file:
        reader = csv.DictReader(csv_file,
                                delimiter=str(delimiter),
                                fieldnames=fields)
        if len(reader.fieldnames) == 1:
            raise RuntimeError(('A "CSV" spectral data file should define '
                                'the following fields: '
                                '("wavelength", "field 1", ..., "field n")!'))

        wavelength = reader.fieldnames[0]
        fields = reader.fieldnames[1:]

        data = OrderedDict(zip(fields, ({} for _ in range(len(fields)))))
        for line in reader:
            for field in fields:
                try:
                    value = np.float_(line[field])
                except ValueError:
                    value = default

                data[field][np.float_(line[wavelength])] = value
        return data


def read_spds_from_csv_file(path,
                            delimiter=',',
                            fields=None,
                            default=0):
    """
    Reads the spectral data from given *CSV* file and return its content as an
    *OrderedDict* of
    :class:`colour.colorimetry.spectrum.SpectralPowerDistribution` classes.

    Parameters
    ----------
    path : unicode
        Absolute *CSV* file path.
    delimiter : unicode, optional
        *CSV* file content delimiter.
    fields : array_like, optional
        *CSV* file spectral data fields names. If no value is provided the
        first line of the file will be used for as spectral data fields names.
    default : numeric
        Default value for fields row with missing value.

    Returns
    -------
    OrderedDict
        :class:`colour.colorimetry.spectrum.SpectralPowerDistribution`
        classes of given *CSV* file.

    Examples
    --------
    >>> import os
    >>> csv_file = os.path.join(
    ...     os.path.dirname(__file__),
    ...     'tests',
    ...     'resources',
    ...     'colorchecker_n_ohta.csv')
    >>> spds = read_spds_from_csv_file(csv_file)
    >>> print(tuple(spds.keys()))
    ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', \
'14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24')
    >>> print(spds['1'])
    SpectralPowerDistribution('1', (380.0, 780.0, 5.0))
    """

    data = read_spectral_data_from_csv_file(path,
                                            delimiter,
                                            fields,
                                            default)

    spds = OrderedDict(((key, SpectralPowerDistribution(key, value))
                        for key, value in data.items()))
    return spds


def write_spds_to_csv_file(spds,
                           path,
                           delimiter=',',
                           fields=None):
    """
    Writes the given spectral power distributions to given *CSV* file.

    Parameters
    ----------
    spds : dict
        Spectral power distributions to write.
    path : unicode
        Absolute *CSV* file path.
    delimiter : unicode, optional
        *CSV* file content delimiter.
    fields : array_like, optional
        *CSV* file spectral data fields names. If no value is provided the
        order of fields will be the one defined by the sorted spectral power
        distributions *dict*.

    Returns
    -------
    bool
        Definition success.

    Raises
    ------
    RuntimeError
        If the given spectral power distributions have different shapes.
    """

    if len(spds) != 1:
        shapes = [spd.shape for spd in spds.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise RuntimeError(('Cannot write spectral power distributions '
                                'with different shapes to "CSV" file!'))

    wavelengths = tuple(spds.values())[0].wavelengths
    with open(path, 'w') as csv_file:
        fields = list(fields) if fields is not None else sorted(spds.keys())
        writer = csv.DictWriter(csv_file,
                                delimiter=str(delimiter),
                                fieldnames=['wavelength'] + fields,
                                lineterminator='\n')
        # Python 2.7.x / 3.4.x only.
        # writer.writeheader()
        writer.writerow(dict((name, name) for name in writer.fieldnames))
        for wavelength in wavelengths:
            row = {'wavelength': wavelength}
            row.update(
                dict((field, spds[field][wavelength]) for field in fields))
            writer.writerow(row)

    return True
