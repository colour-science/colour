"""
UPRTek and Sekonic Spectral Data
================================

Defines the input and output objects for *UPRTek* and *Sekonic*
*Pseudo-XLS*/*CSV* spectral data files.

-   :class:`colour.SpectralDistribution_UPRTek`
-   :class:`colour.SpectralDistribution_Sekonic`
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

from colour.io import SpectralDistribution_IESTM2714
from colour.hints import Any, Dict, List, cast
from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SpectralDistribution_UPRTek",
    "SpectralDistribution_Sekonic",
]


class SpectralDistribution_UPRTek(SpectralDistribution_IESTM2714):
    """
    Implement support to read and write *IES TM-27-14* spectral data XML file
    from a *UPRTek* *Pseudo-XLS* file.

    Parameters
    ----------
    path
        Path for *UPRTek* *Pseudo-XLS* file.

    Attributes
    ----------
    -   :attr:`~colour.SpectralDistribution_UPRTek.metadata`

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution_UPRTek.__init__`
    -   :meth:`~colour.SpectralDistribution_UPRTek.read`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> from colour import SpectralShape
    >>> directory = join(dirname(__file__), 'tests', 'resources')
    >>> sd = SpectralDistribution_UPRTek(
    ...     join(directory, 'ESPD2021_0104_231446.xls'))
    >>> print(sd.read().align(SpectralShape(380, 780, 10)))
    [[  3.80000000e+02   3.02670000e-02]
     [  3.90000000e+02   3.52230000e-02]
     [  4.00000000e+02   1.93250000e-02]
     [  4.10000000e+02   2.94260000e-02]
     [  4.20000000e+02   8.76780000e-02]
     [  4.30000000e+02   6.32578000e-01]
     [  4.40000000e+02   3.62565900e+00]
     [  4.50000000e+02   1.42069180e+01]
     [  4.60000000e+02   1.70112970e+01]
     [  4.70000000e+02   1.19673130e+01]
     [  4.80000000e+02   8.42736200e+00]
     [  4.90000000e+02   7.97729800e+00]
     [  5.00000000e+02   8.71903600e+00]
     [  5.10000000e+02   9.55321500e+00]
     [  5.20000000e+02   9.90610500e+00]
     [  5.30000000e+02   9.91394400e+00]
     [  5.40000000e+02   9.74738000e+00]
     [  5.50000000e+02   9.53404900e+00]
     [  5.60000000e+02   9.27392200e+00]
     [  5.70000000e+02   9.02323400e+00]
     [  5.80000000e+02   8.91788800e+00]
     [  5.90000000e+02   9.11454600e+00]
     [  6.00000000e+02   9.55787100e+00]
     [  6.10000000e+02   1.00600760e+01]
     [  6.20000000e+02   1.04846200e+01]
     [  6.30000000e+02   1.05679540e+01]
     [  6.40000000e+02   1.04359870e+01]
     [  6.50000000e+02   9.82122300e+00]
     [  6.60000000e+02   8.77578300e+00]
     [  6.70000000e+02   7.56471800e+00]
     [  6.80000000e+02   6.29808600e+00]
     [  6.90000000e+02   5.15623400e+00]
     [  7.00000000e+02   4.05390600e+00]
     [  7.10000000e+02   3.06638600e+00]
     [  7.20000000e+02   2.19250000e+00]
     [  7.30000000e+02   1.53922800e+00]
     [  7.40000000e+02   1.14938200e+00]
     [  7.50000000e+02   9.05095000e-01]
     [  7.60000000e+02   6.90947000e-01]
     [  7.70000000e+02   5.08426000e-01]
     [  7.80000000e+02   4.11766000e-01]]
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
    >>> sd.write(join(directory, 'ESPD2021_0104_231446.spdx'))
    ... # doctest: +SKIP
    """

    def __init__(self, path: str, **kwargs: Any):
        super().__init__(path, **kwargs)

        self._delimiter: str = "\t"
        self._spectral_section: str = "380"
        self._spectral_data_pattern: str = "(\\d{3})nm"

        self._metadata: Dict = {}

    @property
    def metadata(self) -> Dict:
        """
        Getter property for the metadata.

        Returns
        -------
        :class:`dict`
            Metadata.
        """

        return self._metadata

    def read(self) -> SpectralDistribution_UPRTek:
        """
        Read and parses the spectral data from a given *UPRTek* *CSV* file.

        Returns
        -------
        :class:`colour.SpectralDistribution_UPRTek`
            *UPRTek* spectral distribution.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_UPRTek(
        ...     join(directory, 'ESPD2021_0104_231446.xls'))
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        [[  3.80000000e+02   3.02670000e-02]
         [  3.90000000e+02   3.52230000e-02]
         [  4.00000000e+02   1.93250000e-02]
         [  4.10000000e+02   2.94260000e-02]
         [  4.20000000e+02   8.76780000e-02]
         [  4.30000000e+02   6.32578000e-01]
         [  4.40000000e+02   3.62565900e+00]
         [  4.50000000e+02   1.42069180e+01]
         [  4.60000000e+02   1.70112970e+01]
         [  4.70000000e+02   1.19673130e+01]
         [  4.80000000e+02   8.42736200e+00]
         [  4.90000000e+02   7.97729800e+00]
         [  5.00000000e+02   8.71903600e+00]
         [  5.10000000e+02   9.55321500e+00]
         [  5.20000000e+02   9.90610500e+00]
         [  5.30000000e+02   9.91394400e+00]
         [  5.40000000e+02   9.74738000e+00]
         [  5.50000000e+02   9.53404900e+00]
         [  5.60000000e+02   9.27392200e+00]
         [  5.70000000e+02   9.02323400e+00]
         [  5.80000000e+02   8.91788800e+00]
         [  5.90000000e+02   9.11454600e+00]
         [  6.00000000e+02   9.55787100e+00]
         [  6.10000000e+02   1.00600760e+01]
         [  6.20000000e+02   1.04846200e+01]
         [  6.30000000e+02   1.05679540e+01]
         [  6.40000000e+02   1.04359870e+01]
         [  6.50000000e+02   9.82122300e+00]
         [  6.60000000e+02   8.77578300e+00]
         [  6.70000000e+02   7.56471800e+00]
         [  6.80000000e+02   6.29808600e+00]
         [  6.90000000e+02   5.15623400e+00]
         [  7.00000000e+02   4.05390600e+00]
         [  7.10000000e+02   3.06638600e+00]
         [  7.20000000e+02   2.19250000e+00]
         [  7.30000000e+02   1.53922800e+00]
         [  7.40000000e+02   1.14938200e+00]
         [  7.50000000e+02   9.05095000e-01]
         [  7.60000000e+02   6.90947000e-01]
         [  7.70000000e+02   5.08426000e-01]
         [  7.80000000e+02   4.11766000e-01]]
        """

        path = cast(str, self.path)

        def as_array(a: Any) -> List:
            """
            Input list of numbers and converts each element to
            float data type.
            """

            return [float(e) for e in a]

        spectral_sections = defaultdict(list)
        with open(path, encoding="utf-8") as csv_file:
            content = csv.reader(csv_file, delimiter=self._delimiter)

            spectral_section = 0
            for row in content:
                if not "".join(row).strip():
                    continue

                attribute, tokens = row[0], row[1:]
                value = tokens[0] if len(tokens) == 1 else tokens

                match = re.match(self._spectral_data_pattern, attribute)
                if match:
                    wavelength = match.group(1)

                    if wavelength == self._spectral_section:
                        spectral_section += 1

                    spectral_sections[spectral_section].append(
                        [wavelength, value]
                    )
                else:
                    for method in (int, float, as_array):
                        try:
                            self._metadata[attribute] = method(
                                value
                            )  # type: ignore[operator]
                            break
                        except Exception:
                            self._metadata[attribute] = value

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
    Implement support to read and write *IES TM-27-14* spectral data XML file
    from a *Sekonic* *CSV* file.

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
    -   :meth:`~colour.SpectralDistribution_Sekonic.read`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> from colour import SpectralShape
    >>> directory = join(dirname(__file__), 'tests', 'resources')
    >>> sd = SpectralDistribution_Sekonic(
    ...     join(directory, 'RANDOM_001_02._3262K.csv'))
    >>> print(sd.read().align(SpectralShape(380, 780, 10)))
    [[  3.80000000e+02   1.69406589e-21]
     [  3.90000000e+02   2.11758237e-22]
     [  4.00000000e+02   1.19813650e-05]
     [  4.10000000e+02   1.97110530e-05]
     [  4.20000000e+02   2.99661440e-05]
     [  4.30000000e+02   6.38192720e-05]
     [  4.40000000e+02   1.68909683e-04]
     [  4.50000000e+02   3.31902935e-04]
     [  4.60000000e+02   3.33143020e-04]
     [  4.70000000e+02   2.30227481e-04]
     [  4.80000000e+02   1.66981976e-04]
     [  4.90000000e+02   1.64439844e-04]
     [  5.00000000e+02   2.01534538e-04]
     [  5.10000000e+02   2.57840526e-04]
     [  5.20000000e+02   3.04612651e-04]
     [  5.30000000e+02   3.41368344e-04]
     [  5.40000000e+02   3.63639323e-04]
     [  5.50000000e+02   3.87050648e-04]
     [  5.60000000e+02   4.21619130e-04]
     [  5.70000000e+02   4.58150520e-04]
     [  5.80000000e+02   5.01176575e-04]
     [  5.90000000e+02   5.40883630e-04]
     [  6.00000000e+02   5.71256795e-04]
     [  6.10000000e+02   5.83703280e-04]
     [  6.20000000e+02   5.57688472e-04]
     [  6.30000000e+02   5.17328095e-04]
     [  6.40000000e+02   4.39994939e-04]
     [  6.50000000e+02   3.62766819e-04]
     [  6.60000000e+02   2.96465587e-04]
     [  6.70000000e+02   2.43966802e-04]
     [  6.80000000e+02   2.04134776e-04]
     [  6.90000000e+02   1.75304012e-04]
     [  7.00000000e+02   1.52887544e-04]
     [  7.10000000e+02   1.29795619e-04]
     [  7.20000000e+02   1.03122693e-04]
     [  7.30000000e+02   8.77607820e-05]
     [  7.40000000e+02   7.61524130e-05]
     [  7.50000000e+02   7.06516880e-05]
     [  7.60000000e+02   3.72199210e-05]
     [  7.70000000e+02   3.63058860e-05]
     [  7.80000000e+02   3.55755470e-05]]
    >>> sd.header.comments # doctest: +SKIP
    >>> sd.metadata.keys() # doctest: +SKIP
    >>> sd.write(join(directory, 'RANDOM_001_02._3262K.spdx')
    ... # doctest: +SKIP
    """

    def __init__(self, path: str, **kwargs: Any):
        super().__init__(path, **kwargs)

        self._delimiter: str = ","
        self._spectral_section: str = "380"
        self._spectral_data_pattern: str = "Spectral Data (\\d{3})\\[nm\\]"

    def read(self) -> SpectralDistribution_Sekonic:
        """
        Read and parses the spectral data from a given *Sekonic* *Pseudo-XLS*
        file.

        Returns
        -------
        :class:`colour.SpectralDistribution_Sekonic`
            *Sekonic* spectral distribution.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from colour import SpectralShape
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_Sekonic(
        ...     join(directory, 'RANDOM_001_02._3262K.csv'))
        >>> print(sd.read().align(SpectralShape(380, 780, 10)))
        [[  3.80000000e+02   1.69406589e-21]
         [  3.90000000e+02   2.11758237e-22]
         [  4.00000000e+02   1.19813650e-05]
         [  4.10000000e+02   1.97110530e-05]
         [  4.20000000e+02   2.99661440e-05]
         [  4.30000000e+02   6.38192720e-05]
         [  4.40000000e+02   1.68909683e-04]
         [  4.50000000e+02   3.31902935e-04]
         [  4.60000000e+02   3.33143020e-04]
         [  4.70000000e+02   2.30227481e-04]
         [  4.80000000e+02   1.66981976e-04]
         [  4.90000000e+02   1.64439844e-04]
         [  5.00000000e+02   2.01534538e-04]
         [  5.10000000e+02   2.57840526e-04]
         [  5.20000000e+02   3.04612651e-04]
         [  5.30000000e+02   3.41368344e-04]
         [  5.40000000e+02   3.63639323e-04]
         [  5.50000000e+02   3.87050648e-04]
         [  5.60000000e+02   4.21619130e-04]
         [  5.70000000e+02   4.58150520e-04]
         [  5.80000000e+02   5.01176575e-04]
         [  5.90000000e+02   5.40883630e-04]
         [  6.00000000e+02   5.71256795e-04]
         [  6.10000000e+02   5.83703280e-04]
         [  6.20000000e+02   5.57688472e-04]
         [  6.30000000e+02   5.17328095e-04]
         [  6.40000000e+02   4.39994939e-04]
         [  6.50000000e+02   3.62766819e-04]
         [  6.60000000e+02   2.96465587e-04]
         [  6.70000000e+02   2.43966802e-04]
         [  6.80000000e+02   2.04134776e-04]
         [  6.90000000e+02   1.75304012e-04]
         [  7.00000000e+02   1.52887544e-04]
         [  7.10000000e+02   1.29795619e-04]
         [  7.20000000e+02   1.03122693e-04]
         [  7.30000000e+02   8.77607820e-05]
         [  7.40000000e+02   7.61524130e-05]
         [  7.50000000e+02   7.06516880e-05]
         [  7.60000000e+02   3.72199210e-05]
         [  7.70000000e+02   3.63058860e-05]
         [  7.80000000e+02   3.55755470e-05]]
        """

        super().read()

        self.header.report_date = self._metadata.get("Date Saved")
        self.header.manufacturer = "Sekonic"

        return self
