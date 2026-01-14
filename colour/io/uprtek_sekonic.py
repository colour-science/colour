"""
UPRTek and Sekonic Spectral Data
================================

Define input and output objects for parsing and handling *UPRTek* and
*Sekonic* spectral measurement data stored in *Pseudo-XLS* and *CSV* file
formats.

-   :class:`colour.SpectralDistribution_UPRTek`
-   :class:`colour.SpectralDistribution_Sekonic`
"""

from __future__ import annotations

import csv
import json
import os
import re
import typing
from collections import defaultdict

if typing.TYPE_CHECKING:
    from colour.hints import Any, PathLike

from colour.hints import cast
from colour.io import SpectralDistribution_IESTM2714
from colour.utilities import as_float_array, as_float_scalar

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SpectralDistribution_UPRTek",
    "SpectralDistribution_Sekonic",
]


class SpectralDistribution_UPRTek(SpectralDistribution_IESTM2714):
    """
    Implement support to read and write *IES TM-27-14* spectral data XML
    files from *UPRTek* *Pseudo-XLS* files.

    This class extends :class:`SpectralDistribution_IESTM2714` to handle
    the specific *Pseudo-XLS* format used by *UPRTek* spectral measurement
    devices. The implementation parses metadata embedded in the file and
    converts it to the standard *IES TM-27-14* XML format.

    Parameters
    ----------
    path
        Absolute or relative path to the *UPRTek* *Pseudo-XLS* file.

    Attributes
    ----------
    -   :attr:`~colour.SpectralDistribution_UPRTek.metadata`

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution_UPRTek.__init__`
    -   :meth:`~colour.SpectralDistribution_UPRTek.__str__`
    -   :meth:`~colour.SpectralDistribution_UPRTek.read`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> from colour import SpectralShape
    >>> directory = join(dirname(__file__), "tests", "resources")
    >>> sd = SpectralDistribution_UPRTek(
    ...     join(directory, "ESPD2021_0104_231446.xls")
    ... )
    >>> print(sd.align(SpectralShape(380, 780, 10)))
    ... # doctest: +ELLIPSIS
    UPRTek
    ======
    <BLANKLINE>
    Path                  : ...
    Spectral Quantity     : irradiance
    Reflection Geometry   : None
    Transmission Geometry : None
    Bandwidth (FWHM)      : None
    Bandwidth Corrected   : None
    <BLANKLINE>
    Header
    ------
    <BLANKLINE>
    Manufacturer           : UPRTek
    Catalog Number         : None
    Description            : None
    Document Creator       : None
    Unique Identifier      : None
    Measurement Equipment  : CV600
    Laboratory             : None
    Report Number          : None
    Report Date            : 2021/01/04_23:14:46
    Document Creation Date : None
    Comments               : {...}
    <BLANKLINE>
    Spectral Data
    -------------
    <BLANKLINE>
    [[3.8000000e+02 3.0267000e-02]
     [3.9000000e+02 3.5223000e-02]
     [4.0000000e+02 1.9325000e-02]
     [4.1000000e+02 2.9426000e-02]
     [4.2000000e+02 8.7678000e-02]
     [4.3000000e+02 6.3257800e-01]
     [4.4000000e+02 3.6256590e+00]
     [4.5000000e+02 1.4206918e+01]
     [4.6000000e+02 1.7011297e+01]
     [4.7000000e+02 1.1967313e+01]
     [4.8000000e+02 8.4273620e+00]
     [4.9000000e+02 7.9772980e+00]
     [5.0000000e+02 8.7190360e+00]
     [5.1000000e+02 9.5532150e+00]
     [5.2000000e+02 9.9061050e+00]
     [5.3000000e+02 9.9139440e+00]
     [5.4000000e+02 9.7473800e+00]
     [5.5000000e+02 9.5340490e+00]
     [5.6000000e+02 9.2739220e+00]
     [5.7000000e+02 9.0232340e+00]
     [5.8000000e+02 8.9178880e+00]
     [5.9000000e+02 9.1145460e+00]
     [6.0000000e+02 9.5578710e+00]
     [6.1000000e+02 1.0060076e+01]
     [6.2000000e+02 1.0484620e+01]
     [6.3000000e+02 1.0567954e+01]
     [6.4000000e+02 1.0435987e+01]
     [6.5000000e+02 9.8212230e+00]
     [6.6000000e+02 8.7757830e+00]
     [6.7000000e+02 7.5647180e+00]
     [6.8000000e+02 6.2980860e+00]
     [6.9000000e+02 5.1562340e+00]
     [7.0000000e+02 4.0539060e+00]
     [7.1000000e+02 3.0663860e+00]
     [7.2000000e+02 2.1925000e+00]
     [7.3000000e+02 1.5392280e+00]
     [7.4000000e+02 1.1493820e+00]
     [7.5000000e+02 9.0509500e-01]
     [7.6000000e+02 6.9094700e-01]
     [7.7000000e+02 5.0842600e-01]
     [7.8000000e+02 4.1176600e-01]]

    >>> sd.header.comments
    '{"Model Name": "CV600", "Serial Number": "19J00789", \
"Time": "2021/01/04_23:14:46", "Memo": [], "LUX": 695.154907, \
"fc": 64.605476, "CCT": 5198.0, "Duv": -0.00062, "I-Time": 12000.0, \
"X": 682.470886, "Y": 695.154907, "Z": 631.635071, "x": 0.339663, \
"y": 0.345975, "u\\'": 0.209915, "v\\'": 0.481087, "LambdaP": 456.0, \
"LambdaPValue": 18.404581, "CRI": 92.956993, "R1": 91.651062, \
"R2": 93.014732, "R3": 97.032013, "R4": 93.513229, "R5": 92.48259, \
"R6": 91.48687, "R7": 93.016129, "R8": 91.459312, "R9": 77.613075, \
"R10": 86.981613, "R11": 94.841324, "R12": 74.139542, "R13": 91.073837, \
"R14": 97.064323, "R15": 88.615669, "TLCI": 97.495056, "TLMF-A": 1.270032, \
"SSI-A": 44.881924, "Rf": 87.234917, "Rg": 98.510712, "IRR": 2.607891}'
    >>> sd.metadata.keys()
    dict_keys(['Model Name', 'Serial Number', 'Time', 'Memo', 'LUX', 'fc', \
'CCT', 'Duv', 'I-Time', 'X', 'Y', 'Z', 'x', 'y', "u'", "v'", 'LambdaP', \
'LambdaPValue', 'CRI', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', \
'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'TLCI', 'TLMF-A', 'SSI-A', 'Rf', \
'Rg', 'IRR'])
    >>> sd.write(join(directory, "ESPD2021_0104_231446.spdx"))
    ... # doctest: +SKIP
    """

    _DELIMITER: str = "\t"
    _SPECTRAL_SECTION: str = "380"
    _SPECTRAL_DATA_PATTERN: str = "(\\d{3})nm"

    def __init__(self, path: str | PathLike, **kwargs: Any) -> None:
        self._metadata: dict = {}

        super().__init__(path, **kwargs)

    @property
    def metadata(self) -> dict:
        """
        Getter for the dataset metadata.

        Returns
        -------
        :class:`dict`
            Dataset metadata containing information about the data source,
            structure, and properties.
        """

        return self._metadata

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *UPRTek* spectral
        distribution.

        Returns
        -------
        :class:`str`
            Formatted string representation.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_UPRTek(
        ...     join(directory, "ESPD2021_0104_231446.xls")
        ... )
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        ... # doctest: +ELLIPSIS
        UPRTek
        ======
        <BLANKLINE>
        Path                  : ...
        Spectral Quantity     : irradiance
        Reflection Geometry   : None
        Transmission Geometry : None
        Bandwidth (FWHM)      : None
        Bandwidth Corrected   : None
        <BLANKLINE>
        Header
        ------
        <BLANKLINE>
        Manufacturer           : UPRTek
        Catalog Number         : None
        Description            : None
        Document Creator       : None
        Unique Identifier      : None
        Measurement Equipment  : CV600
        Laboratory             : None
        Report Number          : None
        Report Date            : 2021/01/04_23:14:46
        Document Creation Date : None
        Comments               : {...}
        <BLANKLINE>
        Spectral Data
        -------------
        <BLANKLINE>
        [[3.8000000e+02 3.0267000e-02]
         [3.9000000e+02 3.5223000e-02]
         [4.0000000e+02 1.9325000e-02]
         [4.1000000e+02 2.9426000e-02]
         [4.2000000e+02 8.7678000e-02]
         [4.3000000e+02 6.3257800e-01]
         [4.4000000e+02 3.6256590e+00]
         [4.5000000e+02 1.4206918e+01]
         [4.6000000e+02 1.7011297e+01]
         [4.7000000e+02 1.1967313e+01]
         [4.8000000e+02 8.4273620e+00]
         [4.9000000e+02 7.9772980e+00]
         [5.0000000e+02 8.7190360e+00]
         [5.1000000e+02 9.5532150e+00]
         [5.2000000e+02 9.9061050e+00]
         [5.3000000e+02 9.9139440e+00]
         [5.4000000e+02 9.7473800e+00]
         [5.5000000e+02 9.5340490e+00]
         [5.6000000e+02 9.2739220e+00]
         [5.7000000e+02 9.0232340e+00]
         [5.8000000e+02 8.9178880e+00]
         [5.9000000e+02 9.1145460e+00]
         [6.0000000e+02 9.5578710e+00]
         [6.1000000e+02 1.0060076e+01]
         [6.2000000e+02 1.0484620e+01]
         [6.3000000e+02 1.0567954e+01]
         [6.4000000e+02 1.0435987e+01]
         [6.5000000e+02 9.8212230e+00]
         [6.6000000e+02 8.7757830e+00]
         [6.7000000e+02 7.5647180e+00]
         [6.8000000e+02 6.2980860e+00]
         [6.9000000e+02 5.1562340e+00]
         [7.0000000e+02 4.0539060e+00]
         [7.1000000e+02 3.0663860e+00]
         [7.2000000e+02 2.1925000e+00]
         [7.3000000e+02 1.5392280e+00]
         [7.4000000e+02 1.1493820e+00]
         [7.5000000e+02 9.0509500e-01]
         [7.6000000e+02 6.9094700e-01]
         [7.7000000e+02 5.0842600e-01]
         [7.8000000e+02 4.1176600e-01]]
        """

        representation = super().__str__()

        return representation.replace(
            ("IES TM-27-14 Spectral Distribution\n=================================="),
            "UPRTek\n======",
        )

    def read(self) -> SpectralDistribution_UPRTek:
        """
        Read and parse the spectral data from the specified *UPRTek* *CSV*
        file.

        Returns
        -------
        :class:`colour.SpectralDistribution_UPRTek`
            *UPRTek* spectral distribution.

        Raises
        ------
        IOError
            If the file cannot be read.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_UPRTek(
        ...     join(directory, "ESPD2021_0104_231446.xls")
        ... )
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        ... # doctest: +ELLIPSIS
        UPRTek
        ======
        <BLANKLINE>
        Path                  : ...
        Spectral Quantity     : irradiance
        Reflection Geometry   : None
        Transmission Geometry : None
        Bandwidth (FWHM)      : None
        Bandwidth Corrected   : None
        <BLANKLINE>
        Header
        ------
        <BLANKLINE>
        Manufacturer           : UPRTek
        Catalog Number         : None
        Description            : None
        Document Creator       : None
        Unique Identifier      : None
        Measurement Equipment  : CV600
        Laboratory             : None
        Report Number          : None
        Report Date            : 2021/01/04_23:14:46
        Document Creation Date : None
        Comments               : {...}
        <BLANKLINE>
        Spectral Data
        -------------
        <BLANKLINE>
        [[3.8000000e+02 3.0267000e-02]
         [3.9000000e+02 3.5223000e-02]
         [4.0000000e+02 1.9325000e-02]
         [4.1000000e+02 2.9426000e-02]
         [4.2000000e+02 8.7678000e-02]
         [4.3000000e+02 6.3257800e-01]
         [4.4000000e+02 3.6256590e+00]
         [4.5000000e+02 1.4206918e+01]
         [4.6000000e+02 1.7011297e+01]
         [4.7000000e+02 1.1967313e+01]
         [4.8000000e+02 8.4273620e+00]
         [4.9000000e+02 7.9772980e+00]
         [5.0000000e+02 8.7190360e+00]
         [5.1000000e+02 9.5532150e+00]
         [5.2000000e+02 9.9061050e+00]
         [5.3000000e+02 9.9139440e+00]
         [5.4000000e+02 9.7473800e+00]
         [5.5000000e+02 9.5340490e+00]
         [5.6000000e+02 9.2739220e+00]
         [5.7000000e+02 9.0232340e+00]
         [5.8000000e+02 8.9178880e+00]
         [5.9000000e+02 9.1145460e+00]
         [6.0000000e+02 9.5578710e+00]
         [6.1000000e+02 1.0060076e+01]
         [6.2000000e+02 1.0484620e+01]
         [6.3000000e+02 1.0567954e+01]
         [6.4000000e+02 1.0435987e+01]
         [6.5000000e+02 9.8212230e+00]
         [6.6000000e+02 8.7757830e+00]
         [6.7000000e+02 7.5647180e+00]
         [6.8000000e+02 6.2980860e+00]
         [6.9000000e+02 5.1562340e+00]
         [7.0000000e+02 4.0539060e+00]
         [7.1000000e+02 3.0663860e+00]
         [7.2000000e+02 2.1925000e+00]
         [7.3000000e+02 1.5392280e+00]
         [7.4000000e+02 1.1493820e+00]
         [7.5000000e+02 9.0509500e-01]
         [7.6000000e+02 6.9094700e-01]
         [7.7000000e+02 5.0842600e-01]
         [7.8000000e+02 4.1176600e-01]]
        """

        path = cast("str", self.path)

        def as_array(a: Any) -> list:
            """
            Input list of numbers and converts each element to
            float data type.
            """

            return [as_float_scalar(e) for e in a]

        spectral_sections = defaultdict(list)
        with open(path, encoding="utf-8") as csv_file:
            content = csv.reader(csv_file, delimiter=self._DELIMITER)

            spectral_section = 0
            for row in content:
                if not "".join(row).strip():
                    continue

                attribute, tokens = row[0], row[1:]
                value = tokens[0] if len(tokens) == 1 else tokens

                if (
                    match := re.match(self._SPECTRAL_DATA_PATTERN, attribute)
                ) is not None:
                    wavelength = match.group(1)

                    if wavelength == self._SPECTRAL_SECTION:
                        spectral_section += 1

                    spectral_sections[spectral_section].append([wavelength, value])
                else:
                    for method in (int, float, as_array):
                        try:
                            self._metadata[attribute] = method(
                                value  # pyright: ignore
                            )
                        except (TypeError, ValueError):
                            self._metadata[attribute] = value
                        else:
                            break

        self.name = os.path.splitext(os.path.basename(path))[0]
        spectral_data = as_float_array(
            spectral_sections[sorted(spectral_sections.keys())[-1]]
        )

        self.wavelengths = spectral_data[..., 0]
        self.values = spectral_data[..., 1]

        self.header.comments = json.dumps(self._metadata)

        self.header.report_date = self._metadata.get("Time")
        self.header.measurement_equipment = self._metadata.get("Model Name")
        self.header.manufacturer = "UPRTek"
        self.spectral_quantity = "irradiance"

        return self


class SpectralDistribution_Sekonic(SpectralDistribution_UPRTek):
    """
    Provide support for reading and writing *IES TM-27-14* spectral data XML
    files from *Sekonic* *CSV* files.

    This class extends the *UPRTek* spectral distribution functionality to
    handle *Sekonic* spectrometer data files. It enables conversion between
    *Sekonic* *CSV* format and the standardized *IES TM-27-14* XML format.

    Parameters
    ----------
    path
        Path for *Sekonic* *CSV* file.

    Attributes
    ----------
    -   :attr:`~colour.SpectralDistribution_UPRTek.metadata`

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution_Sekonic.__init__`
    -   :meth:`~colour.SpectralDistribution_Sekonic.__str__`
    -   :meth:`~colour.SpectralDistribution_Sekonic.read`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> from colour import SpectralShape
    >>> directory = join(dirname(__file__), "tests", "resources")
    >>> sd = SpectralDistribution_Sekonic(join(directory, "RANDOM_001_02._3262K.csv"))
    >>> print(sd.read().align(SpectralShape(380, 780, 10)))
    ... # doctest: +ELLIPSIS
    Sekonic
    =======
    <BLANKLINE>
    Path                  : ...
    Spectral Quantity     : irradiance
    Reflection Geometry   : None
    Transmission Geometry : None
    Bandwidth (FWHM)      : None
    Bandwidth Corrected   : None
    <BLANKLINE>
    Header
    ------
    <BLANKLINE>
    Manufacturer           : Sekonic
    Catalog Number         : None
    Description            : None
    Document Creator       : None
    Unique Identifier      : None
    Measurement Equipment  : None
    Laboratory             : None
    Report Number          : None
    Report Date            : 15/03/2021 3:44:14 p.m.
    Document Creation Date : None
    Comments               : {...}
    <BLANKLINE>
    Spectral Data
    -------------
    <BLANKLINE>
    [[3.80000000e+02 1.69406589e-21]
     [3.90000000e+02 2.11758237e-22]
     [4.00000000e+02 1.19813650e-05]
     [4.10000000e+02 1.97110530e-05]
     [4.20000000e+02 2.99661440e-05]
     [4.30000000e+02 6.38192720e-05]
     [4.40000000e+02 1.68909683e-04]
     [4.50000000e+02 3.31902935e-04]
     [4.60000000e+02 3.33143020e-04]
     [4.70000000e+02 2.30227481e-04]
     [4.80000000e+02 1.66981976e-04]
     [4.90000000e+02 1.64439844e-04]
     [5.00000000e+02 2.01534538e-04]
     [5.10000000e+02 2.57840526e-04]
     [5.20000000e+02 3.04612651e-04]
     [5.30000000e+02 3.41368344e-04]
     [5.40000000e+02 3.63639323e-04]
     [5.50000000e+02 3.87050648e-04]
     [5.60000000e+02 4.21619130e-04]
     [5.70000000e+02 4.58150520e-04]
     [5.80000000e+02 5.01176575e-04]
     [5.90000000e+02 5.40883630e-04]
     [6.00000000e+02 5.71256795e-04]
     [6.10000000e+02 5.83703280e-04]
     [6.20000000e+02 5.57688472e-04]
     [6.30000000e+02 5.17328095e-04]
     [6.40000000e+02 4.39994939e-04]
     [6.50000000e+02 3.62766819e-04]
     [6.60000000e+02 2.96465587e-04]
     [6.70000000e+02 2.43966802e-04]
     [6.80000000e+02 2.04134776e-04]
     [6.90000000e+02 1.75304012e-04]
     [7.00000000e+02 1.52887544e-04]
     [7.10000000e+02 1.29795619e-04]
     [7.20000000e+02 1.03122693e-04]
     [7.30000000e+02 8.77607820e-05]
     [7.40000000e+02 7.61524130e-05]
     [7.50000000e+02 7.06516880e-05]
     [7.60000000e+02 3.72199210e-05]
     [7.70000000e+02 3.63058860e-05]
     [7.80000000e+02 3.55755470e-05]]
    >>> sd.header.comments  # doctest: +SKIP
    >>> sd.metadata.keys()  # doctest: +SKIP
    >>> sd.write(join(directory, "RANDOM_001_02._3262K.spdx"))
    ... # doctest: +SKIP
    """

    _DELIMITER: str = ","
    _SPECTRAL_SECTION: str = "380"
    _SPECTRAL_DATA_PATTERN: str = "Spectral Data (\\d{3})\\[nm\\]"

    def __init__(self, path: str | PathLike, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *Sekonic* spectral
        distribution.

        Returns
        -------
        :class:`str`
            Formatted string representation.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_UPRTek(
        ...     join(directory, "ESPD2021_0104_231446.xls")
        ... )
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        ... # doctest: +ELLIPSIS
        UPRTek
        ======
        <BLANKLINE>
        Path                  : ...
        Spectral Quantity     : irradiance
        Reflection Geometry   : None
        Transmission Geometry : None
        Bandwidth (FWHM)      : None
        Bandwidth Corrected   : None
        <BLANKLINE>
        Header
        ------
        <BLANKLINE>
        Manufacturer           : UPRTek
        Catalog Number         : None
        Description            : None
        Document Creator       : None
        Unique Identifier      : None
        Measurement Equipment  : CV600
        Laboratory             : None
        Report Number          : None
        Report Date            : 2021/01/04_23:14:46
        Document Creation Date : None
        Comments               : {...}
        <BLANKLINE>
        Spectral Data
        -------------
        <BLANKLINE>
        [[3.8000000e+02 3.0267000e-02]
         [3.9000000e+02 3.5223000e-02]
         [4.0000000e+02 1.9325000e-02]
         [4.1000000e+02 2.9426000e-02]
         [4.2000000e+02 8.7678000e-02]
         [4.3000000e+02 6.3257800e-01]
         [4.4000000e+02 3.6256590e+00]
         [4.5000000e+02 1.4206918e+01]
         [4.6000000e+02 1.7011297e+01]
         [4.7000000e+02 1.1967313e+01]
         [4.8000000e+02 8.4273620e+00]
         [4.9000000e+02 7.9772980e+00]
         [5.0000000e+02 8.7190360e+00]
         [5.1000000e+02 9.5532150e+00]
         [5.2000000e+02 9.9061050e+00]
         [5.3000000e+02 9.9139440e+00]
         [5.4000000e+02 9.7473800e+00]
         [5.5000000e+02 9.5340490e+00]
         [5.6000000e+02 9.2739220e+00]
         [5.7000000e+02 9.0232340e+00]
         [5.8000000e+02 8.9178880e+00]
         [5.9000000e+02 9.1145460e+00]
         [6.0000000e+02 9.5578710e+00]
         [6.1000000e+02 1.0060076e+01]
         [6.2000000e+02 1.0484620e+01]
         [6.3000000e+02 1.0567954e+01]
         [6.4000000e+02 1.0435987e+01]
         [6.5000000e+02 9.8212230e+00]
         [6.6000000e+02 8.7757830e+00]
         [6.7000000e+02 7.5647180e+00]
         [6.8000000e+02 6.2980860e+00]
         [6.9000000e+02 5.1562340e+00]
         [7.0000000e+02 4.0539060e+00]
         [7.1000000e+02 3.0663860e+00]
         [7.2000000e+02 2.1925000e+00]
         [7.3000000e+02 1.5392280e+00]
         [7.4000000e+02 1.1493820e+00]
         [7.5000000e+02 9.0509500e-01]
         [7.6000000e+02 6.9094700e-01]
         [7.7000000e+02 5.0842600e-01]
         [7.8000000e+02 4.1176600e-01]]
        """

        representation = super().__str__()

        return representation.replace(
            "UPRTek\n======",
            "Sekonic\n=======",
        )

    def read(self) -> SpectralDistribution_Sekonic:
        """
        Read and parse the spectral data from the specified *Sekonic*
        *Pseudo-XLS* file.

        Returns
        -------
        :class:`colour.SpectralDistribution_Sekonic`
            *Sekonic* spectral distribution.

        Raises
        ------
        IOError
            If the file cannot be read.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_Sekonic(
        ...     join(directory, "RANDOM_001_02._3262K.csv")
        ... )
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        ... # doctest: +ELLIPSIS
        Sekonic
        =======
        <BLANKLINE>
        Path                  : ...
        Spectral Quantity     : irradiance
        Reflection Geometry   : None
        Transmission Geometry : None
        Bandwidth (FWHM)      : None
        Bandwidth Corrected   : None
        <BLANKLINE>
        Header
        ------
        <BLANKLINE>
        Manufacturer           : Sekonic
        Catalog Number         : None
        Description            : None
        Document Creator       : None
        Unique Identifier      : None
        Measurement Equipment  : None
        Laboratory             : None
        Report Number          : None
        Report Date            : 15/03/2021 3:44:14 p.m.
        Document Creation Date : None
        Comments               : {...}
        <BLANKLINE>
        Spectral Data
        -------------
        <BLANKLINE>
        [[3.80000000e+02 1.69406589e-21]
         [3.90000000e+02 2.11758237e-22]
         [4.00000000e+02 1.19813650e-05]
         [4.10000000e+02 1.97110530e-05]
         [4.20000000e+02 2.99661440e-05]
         [4.30000000e+02 6.38192720e-05]
         [4.40000000e+02 1.68909683e-04]
         [4.50000000e+02 3.31902935e-04]
         [4.60000000e+02 3.33143020e-04]
         [4.70000000e+02 2.30227481e-04]
         [4.80000000e+02 1.66981976e-04]
         [4.90000000e+02 1.64439844e-04]
         [5.00000000e+02 2.01534538e-04]
         [5.10000000e+02 2.57840526e-04]
         [5.20000000e+02 3.04612651e-04]
         [5.30000000e+02 3.41368344e-04]
         [5.40000000e+02 3.63639323e-04]
         [5.50000000e+02 3.87050648e-04]
         [5.60000000e+02 4.21619130e-04]
         [5.70000000e+02 4.58150520e-04]
         [5.80000000e+02 5.01176575e-04]
         [5.90000000e+02 5.40883630e-04]
         [6.00000000e+02 5.71256795e-04]
         [6.10000000e+02 5.83703280e-04]
         [6.20000000e+02 5.57688472e-04]
         [6.30000000e+02 5.17328095e-04]
         [6.40000000e+02 4.39994939e-04]
         [6.50000000e+02 3.62766819e-04]
         [6.60000000e+02 2.96465587e-04]
         [6.70000000e+02 2.43966802e-04]
         [6.80000000e+02 2.04134776e-04]
         [6.90000000e+02 1.75304012e-04]
         [7.00000000e+02 1.52887544e-04]
         [7.10000000e+02 1.29795619e-04]
         [7.20000000e+02 1.03122693e-04]
         [7.30000000e+02 8.77607820e-05]
         [7.40000000e+02 7.61524130e-05]
         [7.50000000e+02 7.06516880e-05]
         [7.60000000e+02 3.72199210e-05]
         [7.70000000e+02 3.63058860e-05]
         [7.80000000e+02 3.55755470e-05]]
        """

        super().read()

        self.header.report_date = self._metadata.get("Date Saved")
        self.header.manufacturer = "Sekonic"

        return self
