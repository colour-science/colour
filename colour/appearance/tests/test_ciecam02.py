from colour.appearance.ciecam02 import XYZ_to_CIECAM02, CIECAM02_SURROUND_FCNC
from colour.appearance.tests.common import ColorAppearanceTest
import numpy as np


class TestCIECAM02ColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'ciecam02.csv'
    output_parameter_dict = {'J': 'J',
                             'Q': 'Q',
                             'C': 'C',
                             'M': 'M',
                             'S': 's'}

    input_parameter_dict = {}

    def create_model_from_data(self, data):
        model = XYZ_to_CIECAM02(np.array([data['X'], data['Y'], data['Z']]),
                                np.array([data['X_W'], data['Y_W'], data['Z_W']]),
                                data['L_A'],
                                data['Y_b'],
                                CIECAM02_SURROUND_FCNC(data['F'], data['c'], data['N_c']),
                                )
        return model