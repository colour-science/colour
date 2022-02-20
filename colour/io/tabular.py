"""
CSV Tabular Data Input / Output
===============================

Defines various input / output objects for *CSV* tabular data files:

-   :func:`colour.read_spectral_data_from_csv_file`
-   :func:`colour.read_sds_from_csv_file`
-   :func:`colour.write_sds_to_csv_file`
"""

from __future__ import annotations

import csv
import numpy as np
import os
import tempfile

from colour.colorimetry import SpectralDistribution
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.hints import Any, Boolean, Dict, NDArray
from colour.utilities import filter_kwargs

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "read_spectral_data_from_csv_file",
    "read_sds_from_csv_file",
    "write_sds_to_csv_file",
]


def read_spectral_data_from_csv_file(
    path: str, **kwargs: Any
) -> Dict[str, NDArray]:
    """
    Read the spectral data from given *CSV* file in the following form::

        390,  4.15003E-04,  3.68349E-04,  9.54729E-03
        395,  1.05192E-03,  9.58658E-04,  2.38250E-02
        400,  2.40836E-03,  2.26991E-03,  5.66498E-02
        ...
        830,  9.74306E-07,  9.53411E-08,  0.00000

    and returns it as an *dict* as follows::

        {
            'wavelength': ndarray,
            'field 1': ndarray,
            'field 2': ndarray,
            ...,
            'field n': ndarray
        }

    Parameters
    ----------
    path
        *CSV* file path.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments passed to :func:`numpy.recfromcsv` definition.

    Returns
    -------
    :class:`dict`
        *CSV* file content.

    Notes
    -----
    -   A *CSV* spectral data file should define at least define two fields:
        one for the wavelengths and one for the associated values of one
        spectral distribution.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> csv_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                         'resources', 'colorchecker_n_ohta.csv')
    >>> sds_data = read_spectral_data_from_csv_file(csv_file)
    >>> pprint(list(sds_data.keys()))
    ['wavelength',
     '1',
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

    settings = {
        "case_sensitive": True,
        "deletechars": "",
        "replace_space": " ",
        "dtype": DEFAULT_FLOAT_DTYPE,
    }
    settings.update(**kwargs)

    transpose = settings.get("transpose")
    if transpose:
        delimiter = settings.get("delimiter", ",")
        if settings.get("delimiter") is not None:
            del settings["delimiter"]

        with open(path) as csv_file:
            content = zip(*csv.reader(csv_file, delimiter=delimiter))

        transposed_csv_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False
        )
        path = transposed_csv_file.name
        csv.writer(transposed_csv_file).writerows(content)
        transposed_csv_file.close()

    data = np.recfromcsv(path, **filter_kwargs(np.genfromtxt, **settings))

    if transpose:
        os.unlink(transposed_csv_file.name)

    return {name: data[name] for name in data.dtype.names}


def read_sds_from_csv_file(
    path: str, **kwargs: Any
) -> Dict[str, SpectralDistribution]:
    """
    Read the spectral data from given *CSV* file and returns its content as a
    *dict* of :class:`colour.SpectralDistribution` class instances.

    Parameters
    ----------
    path
        *CSV* file path.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments passed to :func:`numpy.recfromcsv` definition.

    Returns
    -------
    :class:`dict`
        *Dict* of :class:`colour.SpectralDistribution` class instances.

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
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    data = read_spectral_data_from_csv_file(path, **kwargs)

    fields = list(data.keys())
    wavelength_field, sd_fields = fields[0], fields[1:]

    sds = {
        sd_field: SpectralDistribution(
            data[sd_field], data[wavelength_field], name=sd_field
        )
        for sd_field in sd_fields
    }

    return sds


def write_sds_to_csv_file(
    sds: Dict[str, SpectralDistribution], path: str
) -> Boolean:
    """
    Write the given spectral distributions to given *CSV* file.

    Parameters
    ----------
    sds
        Spectral distributions to write to given *CSV* file.
    path
        *CSV* file path.

    Returns
    -------
    :class:`bool`
        Definition success.

    Raises
    ------
    ValueError
        If the given spectral distributions have different shapes.
    """

    if len(sds) != 1:
        shapes = [sd.shape for sd in sds.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(
                "Cannot write spectral distributions "
                'with different shapes to "CSV" file!'
            )

    wavelengths = tuple(sds.values())[0].wavelengths
    with open(path, "w") as csv_file:
        fields = sorted(sds.keys())
        writer = csv.DictWriter(
            csv_file,
            delimiter=",",
            fieldnames=["wavelength"] + fields,
            lineterminator="\n",
        )

        writer.writeheader()

        for wavelength in wavelengths:
            row = {"wavelength": wavelength}
            row.update({field: sds[field][wavelength] for field in fields})
            writer.writerow(row)

    return True
