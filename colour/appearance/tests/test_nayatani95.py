from colour.appearance.nayatani95 import XYZ_to_Nayatani95
from colour.appearance.tests.common import ColorAppearanceTest


class TestNayataniColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'nayatani.csv'

    output_parameter_dict = {'L_star_P': 'L_star_P',
                             'L_star_P': 'L_star_P',
                             'theta': 'theta',
                             'C': 'C',
                             'S': 'S',
                             'B_r': 'B_r',
                             'M': 'M'}

    def create_model_from_data(self, data):
        model = XYZ_to_Nayatani95(data['X'], data['Y'], data['Z'],
                                  data['X_n'], data['Y_n'], data['Z_n'],
                                  data['Y_o'],
                                  data['E_o'],
                                  data['E_or'])
        return model
