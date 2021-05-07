import csv
from collections import OrderedDict
from colour.colorimetry import SpectralDistribution
from colour.io import SpectralDistribution_IESTM2714
import json

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

UPRTEK_FILE_ENCODING = "utf-8"

__all__ = ["UPRTEK_FILE_ENCODING", "uprtek_parser"]


class uprtek_parser(SpectralDistribution_IESTM2714):
    def __init__(self, path):
        self.path = path

        def read_data(path):
            with open(path, encoding=UPRTEK_FILE_ENCODING) as csv_file:
                return list(
                    filter(None, list(csv.reader(csv_file, delimiter="\t"))))

        self.data = read_data(self.path)

    def read_sd(self):
        sd = OrderedDict()
        for line in self.data:
            if line[0].startswith("nm", 3) and len(line[0]) == 5:
                sd[int(line[0][0:3])] = line[1]
        return SpectralDistribution(sd)

    def read_extra_data(self):
        ext_data = {
            "CCT": None,
            "CRI": None,
            "Duv": None,
            "I-Time": None,
            "IRR": None,
            "LUX": None,
            "LambdaP": None,
            "LambdaPValue": None,
            "Memo": None,
            "Model Name": None,
            "R1": None,
            "R10": None,
            "R11": None,
            "R12": None,
            "R13": None,
            "R14": None,
            "R15": None,
            "R2": None,
            "R3": None,
            "R4": None,
            "R5": None,
            "R6": None,
            "R7": None,
            "R8": None,
            "R9": None,
            "Rf": None,
            "Rg": None,
            "SSI-A": None,
            "Serial Number": None,
            "TLCI": None,
            "TLMF-A": None,
            "Time": None,
            "X": None,
            "Y": None,
            "Z": None,
            "fc": None,
            "u'": None,
            "v'": None,
            "x": None,
            "y": None,
        }

        for t in self.data:
            if t[0] in ext_data:
                ext_data[str(t[0])] = t[1]
        return json.dumps(ext_data)
