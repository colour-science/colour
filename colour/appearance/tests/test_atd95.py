from numpy.testing.utils import assert_almost_equal
from colour.appearance.atd95 import XYZ_to_ATD95, xyz_to_lms, calculate_final_response
from colour.appearance.tests.common import ColorAppearanceTest
import numpy as np


class TestATD95ColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'atd.csv'

    output_parameter_dict = {'A_1': 'a_1',
                             'T_1': 't_1',
                             'D_1': 'd_1',
                             'A_2': 'a_2',
                             'T_2': 't_2',
                             'D_2': 'd_2',
                             'Br': 'Br',
                             'C': 'C',
                             'H': 'H'}

    def create_model_from_data(self, data):
        model = XYZ_to_ATD95(data['X'], data['Y'], data['Z'],
                             data['X_0'], data['Y_0'], data['Z_0'],
                             data['Y_02'],
                             data['K_1'], data['K_2'],
                             data['sigma'])
        return model

    def test_xyz_to_lms(self):
        l, m, s = xyz_to_lms(np.array([1, 1, 1]))
        assert_almost_equal(l, 0.7946522478109985)
        assert_almost_equal(m, 0.9303058494144267)
        assert_almost_equal(s, 0.7252006614718631)

    def test_final_response_calculation(self):
        assert_almost_equal(calculate_final_response(0), 0)
        assert_almost_equal(calculate_final_response(100), 1.0 / 3.0)
        assert_almost_equal(calculate_final_response(200), 0.5)
        assert_almost_equal(calculate_final_response(10000), 0.980392157)
