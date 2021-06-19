import csv
import json
import os
import re
from collections import defaultdict

from colour.io import SpectralDistribution_IESTM2714
from colour.utilities import as_float_array, is_string


class SpectralDistribution_UPRTek(SpectralDistribution_IESTM2714):
    """
    This class serves as a parser to read and write *IES TM-27-14*
    spectral data XML file from a *UPRTek* CSV file.

    Parameters
    ----------
    path : unicode
        Path for UPRTek CSV file.

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution_UPRTek.__init__`
    -   :meth:`~colour.SpectralDistribution_UPRTek.read`
    -   :meth:`~colour.SpectralDistribution_UPRTek.write`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> directory = dirname(__file__)
    >>> sd = SpectralDistribution_UPRTek(join(directory, 'upr.xls.txt'))
    >>> print(sd.read())
    [[  3.80000000e+02   3.02670000e-02]
    [  3.81000000e+02   3.02670000e-02]
    [  3.82000000e+02   3.02670000e-02]
    [  3.83000000e+02   2.98220000e-02]
    ...
    ...
    [  7.78000000e+02   4.11766000e-01]
    [  7.79000000e+02   4.11766000e-01]
    [  7.80000000e+02   4.11766000e-01]]
    >>> sd['400']
    0.019325
    >>> sd.header.comments
    {"Model Name": "CV600", "Serial Number": "19J00789",
    "Time": "2021/01/04_23:14:46", "Memo": [], "LUX": 695.154907,
    "fc": 64.605476, "CCT": 5198.0, "Duv": -0.00062, "I-Time": 12000.0,
    "X": 682.470886, "Y": 695.154907, "Z": 631.635071, "x": 0.339663,
    "y": 0.345975, "u'": 0.209915, "v'": 0.481087, "LambdaP": 456.0,
    "LambdaPValue": 18.404581, "CRI": 92.956993, "R1": 91.651062,
    "R2": 93.014732, "R3": 97.032013, "R4": 93.513229, "R5": 92.48259,
    "R6": 91.48687, "R7": 93.016129, "R8": 91.459312, "R9": 77.613075,
    "R10": 86.981613, "R11": 94.841324, "R12": 74.139542, "R13": 91.073837,
    "R14": 97.064323, "R15": 88.615669, "TLCI": 97.495056,
    "TLMF-A": 1.270032, "SSI-A": 44.881924, "Rf": 87.234917, "Rg": 98.510712,
    "IRR": 2.607891}

    >>> sd.write(join(directory, 'sd.spdx'))
    """

    def __init__(self, path, delimiter="\t", spectra_section="380", **kwargs):
        super(SpectralDistribution_UPRTek, self).__init__(path, **kwargs)

        self._delimiter = "\t"
        self.delimiter = delimiter

        self._spectra_section = "380"
        self.spectra_section = spectra_section

        self._metadata = {}

    @property
    def delimiter(self):
        """
        Getter and setter property for the delimiter.

        Parameters
        ----------
        value : unicode
            Value to set the delimiter with.

        Returns
        -------
        unicode
            Delimiter.
        """

        return self._delimiter

    @delimiter.setter
    def delimiter(self, value):
        """
        Setter for the **self.delimiter** property.
        """

        if value is not None:
            assert is_string(
                value
            ), '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                "delimiter", value)

        self._delimiter = value

    @property
    def spectra_section(self):
        """
        Getter and setter property for the spectral section.

        Parameters
        ----------
        value : unicode
            Value to set the spectral section with.

        Returns
        -------
        unicode
            Spectra section.
        """

        return self._spectra_section

    @spectra_section.setter
    def spectra_section(self, value):
        """
        Setter for the **self.spectra_section** property.
        """

        if value is not None:
            assert is_string(
                value
            ), '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                "spectra_section", value)

        self._spectra_section = value

    @property
    def metadata(self):
        """
        Getter and setter property for the metadata.

        Parameters
        ----------
        value : dict
            Value to set the metadata with.

        Returns
        -------
        unicode
            Metadata.
        """

        return self._metadata

    def read(self):
        """
        Reads and parses the spectral data from a given *UPRTek* CSV file.

        Returns
        -------
        bool
            Definition success.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> directory = dirname(__file__)
        >>> sd = SpectralDistribution_UPRTek(join(directory, 'upr.xls.txt'))
        >>> print(sd.read())
        [[  3.80000000e+02   3.02670000e-02]
        [  3.81000000e+02   3.02670000e-02]
        [  3.82000000e+02   3.02670000e-02]
        [  3.83000000e+02   2.98220000e-02]
        ...
        ...
        [  7.78000000e+02   4.11766000e-01]
        [  7.79000000e+02   4.11766000e-01]
        [  7.80000000e+02   4.11766000e-01]]
        >>> sd['400']
        0.019325
        """

        def as_array(a):
            return [float(e) for e in a]

        spectral_sections = defaultdict(list)
        with open(self.path, encoding="utf-8") as csv_file:
            content = csv.reader(csv_file, delimiter=self._delimiter)

            spectral_section = 0
            for row in content:
                if not "".join(row).strip():
                    continue

                key, value = row[0], row[1:]
                value = value[0] if len(value) == 1 else value

                search = re.search("(\\d{3})\\[?nm\\]?", key)
                if search:
                    wavelength = search.group(1)

                    if wavelength == self._spectra_section:
                        spectral_section += 1

                    spectral_sections[spectral_section].append(
                        [wavelength, value])
                else:
                    for method in (int, float, as_array):
                        try:
                            self._metadata[key] = method(value)
                            break
                        except Exception:
                            self._metadata[key] = value

        self.name = os.path.splitext(os.path.basename(self.path))[0]
        spectral_data = as_float_array(spectral_sections[sorted(
            spectral_sections.keys())[-1]])

        self.wavelengths = spectral_data[..., 0]
        self.values = spectral_data[..., 1]

        self.header.comments = json.dumps(self._metadata)
        self.header.report_date = self._metadata["Time"][:10]
        self.header.measurement_equipment = self._metadata["Model Name"]
        self.header.manufacturer = "UPRTek"
        self.spectral_quantity = 'Irradiance'

        return self


class SpectralDistribution_Sekonic(SpectralDistribution_UPRTek):
    def __init__(self, path, delimiter=",", spectra_section="380", **kwargs):
        super(SpectralDistribution_Sekonic, self).__init__(
            path, delimiter, spectra_section, **kwargs)
