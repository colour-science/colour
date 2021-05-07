import csv
from collections import OrderedDict
from colour.colorimetry import SpectralDistribution
from colour.io import SpectralDistribution_IESTM2714
import pandas as pd
import json

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

SEKONIC_FILE_ENCODING = "utf-8"

__all__ = ["SEKONIC_FILE_ENCODING", "sekonic_parser"]


class sekonic_parser(SpectralDistribution_IESTM2714):
    def __init__(self, path):
        self.path = path

        def read_data(path):
            with open(path, encoding=SEKONIC_FILE_ENCODING) as csv_file:
                return list(
                    filter(None, list(csv.reader(csv_file, delimiter=","))))

        self.data = read_data(self.path)

    def read_sd(self):

        sd_5, sd_1 = OrderedDict(), OrderedDict()
        counter = 0
        for line in self.data:
            if line[0] == "Spectral Data 380[nm]":
                counter += 1
            if counter == 1 and line[0].startswith("Spectral Data"):
                sd_5[int(line[0][14:17])] = line[1]
            if counter == 2 and line[0].startswith("Spectral Data"):
                sd_1[int(line[0][14:17])] = line[1]

        sekonic_sd_5 = SpectralDistribution(sd_5)
        sekonic_sd_1 = SpectralDistribution(sd_1)

        return (sekonic_sd_5, sekonic_sd_1)

    def read_extra_data(self):
        ext_data = {
            "Date Saved": None,
            "Title": None,
            "Measuring Mode": None,
            "Viewing Angle [°]": None,
            "Tcp [K]": None,
            "⊿uv": None,
            "Illuminance [lx]": None,
            "Illuminance [fc]": None,
            "Peak Wavelength [nm]": None,
            "Tristimulus Value X": None,
            "Tristimulus Value Y": None,
            "Tristimulus Value Z": None,
            "CIE1931 x": None,
            "CIE1931 y": None,
            "CIE1931 z": None,
            "CIE1976 u'": None,
            "CIE1976 v'": None,
            "Dominant Wavelength [nm]": None,
            "Purity [%]": None,
            "PPFD [umolm⁻²s⁻¹]": None,
            "CRI Ra": None,
            "CRI R1": None,
            "CRI R2": None,
            "CRI R3": None,
            "CRI R4": None,
            "CRI R5": None,
            "CRI R6": None,
            "CRI R7": None,
            "CRI R8": None,
            "CRI R9": None,
            "CRI R10": None,
            "CRI R11": None,
            "CRI R12": None,
            "CRI R13": None,
            "CRI R14": None,
            "CRI R15": None,
            "TM-30 Rf": None,
            "TM-30 Rg": None,
            "SSIt": None,
            "SSId": None,
            "SSI1": None,
            "SSI2": None,
            "TLCI": None,
            "TLMF": None,
        }
        for t in self.data:
            if t[0] in ext_data:
                ext_data[str(t[0])] = t[1]
        json_1 = json.dumps(ext_data)
        index = self.data.index([
            "TM-30 Color Vector Graphic",
            "Reference Illuminant x",
            "Reference Illuminant y",
            "Measured Illuminant x",
            "Measured Illuminant y",
        ])
        table = self.data[index:index + 16]
        df = pd.DataFrame(table)
        json_2 = df.to_json()

        jsonMerged = {**json.loads(json_1), **json.loads(json_2)}

        return json.dumps(jsonMerged)
