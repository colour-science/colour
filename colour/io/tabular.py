# -*- coding: utf-8 -*-
"""
CSV Tabular Data Input / Output
===============================

Defines various input / output objects for *CSV* tabular data files:

-   :func:`colour.read_spectral_data_from_csv_file`
-   :func:`colour.read_sds_from_csv_file`
-   :func:`colour.write_sds_to_csv_file`
"""

from __future__ import division, unicode_literals

from collections import OrderedDict
import csv

from colour.colorimetry import SpectralDistribution
from colour.constants import DEFAULT_FLOAT_DTYPE

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'read_spectral_data_from_csv_file', 'read_sds_from_csv_file',
    'write_sds_to_csv_file'
]


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
        spectral distribution.
    -   If no value is provided for the fields names, the first line of the
        file will be used as spectral data fields names.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> csv_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                         'resources', 'colorchecker_n_ohta.csv')
    >>> sds_data = read_spectral_data_from_csv_file(csv_file)
    >>> pprint(list(sds_data.keys()))
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
        reader = csv.DictReader(
            csv_file, delimiter=str(delimiter), fieldnames=fields)
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
                    value = DEFAULT_FLOAT_DTYPE(line[field])
                except ValueError:
                    value = default

                data[field][DEFAULT_FLOAT_DTYPE(line[wavelength])] = value
        return data


def read_sds_from_csv_file(path, delimiter=',', fields=None, default=0):
    """
    Reads the spectral data from given *CSV* file and return its content as an
    *OrderedDict* of :class:`colour.SpectralDistribution` classes.

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
        :class:`colour.SpectralDistribution` classes of given *CSV* file.

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> import os
    >>> csv_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                         'resources', 'colorchecker_n_ohta.csv')
    >>> sds = read_sds_from_csv_file(csv_file)
    >>> print(tuple(sds.keys()))
    ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', \
'14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24')
    >>> with numpy_print_options(suppress=True):
    ...     sds['1']  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.   ,    0.048],
                          [ 385.   ,    0.051],
                          [ 390.   ,    0.055],
                          [ 395.   ,    0.06 ],
                          [ 400.   ,    0.065],
                          [ 405.   ,    0.068],
                          [ 410.   ,    0.068],
                          [ 415.   ,    0.067],
                          [ 420.   ,    0.064],
                          [ 425.   ,    0.062],
                          [ 430.   ,    0.059],
                          [ 435.   ,    0.057],
                          [ 440.   ,    0.055],
                          [ 445.   ,    0.054],
                          [ 450.   ,    0.053],
                          [ 455.   ,    0.053],
                          [ 460.   ,    0.052],
                          [ 465.   ,    0.052],
                          [ 470.   ,    0.052],
                          [ 475.   ,    0.053],
                          [ 480.   ,    0.054],
                          [ 485.   ,    0.055],
                          [ 490.   ,    0.057],
                          [ 495.   ,    0.059],
                          [ 500.   ,    0.061],
                          [ 505.   ,    0.062],
                          [ 510.   ,    0.065],
                          [ 515.   ,    0.067],
                          [ 520.   ,    0.07 ],
                          [ 525.   ,    0.072],
                          [ 530.   ,    0.074],
                          [ 535.   ,    0.075],
                          [ 540.   ,    0.076],
                          [ 545.   ,    0.078],
                          [ 550.   ,    0.079],
                          [ 555.   ,    0.082],
                          [ 560.   ,    0.087],
                          [ 565.   ,    0.092],
                          [ 570.   ,    0.1  ],
                          [ 575.   ,    0.107],
                          [ 580.   ,    0.115],
                          [ 585.   ,    0.122],
                          [ 590.   ,    0.129],
                          [ 595.   ,    0.134],
                          [ 600.   ,    0.138],
                          [ 605.   ,    0.142],
                          [ 610.   ,    0.146],
                          [ 615.   ,    0.15 ],
                          [ 620.   ,    0.154],
                          [ 625.   ,    0.158],
                          [ 630.   ,    0.163],
                          [ 635.   ,    0.167],
                          [ 640.   ,    0.173],
                          [ 645.   ,    0.18 ],
                          [ 650.   ,    0.188],
                          [ 655.   ,    0.196],
                          [ 660.   ,    0.204],
                          [ 665.   ,    0.213],
                          [ 670.   ,    0.222],
                          [ 675.   ,    0.231],
                          [ 680.   ,    0.242],
                          [ 685.   ,    0.251],
                          [ 690.   ,    0.261],
                          [ 695.   ,    0.271],
                          [ 700.   ,    0.282],
                          [ 705.   ,    0.294],
                          [ 710.   ,    0.305],
                          [ 715.   ,    0.318],
                          [ 720.   ,    0.334],
                          [ 725.   ,    0.354],
                          [ 730.   ,    0.372],
                          [ 735.   ,    0.392],
                          [ 740.   ,    0.409],
                          [ 745.   ,    0.42 ],
                          [ 750.   ,    0.436],
                          [ 755.   ,    0.45 ],
                          [ 760.   ,    0.462],
                          [ 765.   ,    0.465],
                          [ 770.   ,    0.448],
                          [ 775.   ,    0.432],
                          [ 780.   ,    0.421]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})
    """

    data = read_spectral_data_from_csv_file(path, delimiter, fields, default)

    sds = OrderedDict(((key, SpectralDistribution(value, name=key))
                       for key, value in data.items()))
    return sds


def write_sds_to_csv_file(sds, path, delimiter=',', fields=None):
    """
    Writes the given spectral distributions to given *CSV* file.

    Parameters
    ----------
    sds : dict
        Spectral distributions to write.
    path : unicode
        Absolute *CSV* file path.
    delimiter : unicode, optional
        *CSV* file content delimiter.
    fields : array_like, optional
        *CSV* file spectral data fields names. If no value is provided the
        order of fields will be the one defined by the sorted spectral
        distributions *dict*.

    Returns
    -------
    bool
        Definition success.

    Raises
    ------
    RuntimeError
        If the given spectral distributions have different shapes.
    """

    if len(sds) != 1:
        shapes = [sd.shape for sd in sds.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise RuntimeError(('Cannot write spectral distributions '
                                'with different shapes to "CSV" file!'))

    wavelengths = tuple(sds.values())[0].wavelengths
    with open(path, 'w') as csv_file:
        fields = list(fields) if fields is not None else sorted(sds.keys())
        writer = csv.DictWriter(
            csv_file,
            delimiter=str(delimiter),
            fieldnames=['wavelength'] + fields,
            lineterminator='\n')
        # Python 2.7.x / 3.4.x only.
        # writer.writeheader()
        writer.writerow(dict((name, name) for name in writer.fieldnames))
        for wavelength in wavelengths:
            row = {'wavelength': wavelength}
            row.update(
                dict((field, sds[field][wavelength]) for field in fields))
            writer.writerow(row)

    return True
