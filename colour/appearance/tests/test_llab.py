from colour.appearance.llab import XYZ_to_LLAB
from colour.appearance.tests.common import ColorAppearanceTest


class TestLLABColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'llab.csv'

    output_parameter_dict = {'L_L': 'L_L',
                             'Ch_L': 'Ch_L',
                             's_L': 's_L',
                             'h_L': 'h_L',
                             'A_L': 'A_L',
                             'B_L': 'B_L'}

    def create_model_from_data(self, data):
        model = XYZ_to_LLAB(data['X'], data['Y'], data['Z'],
                     data['X_0'], data['Y_0'], data['Z_0'],
                     data['Y_b'],
                     data['F_S'],
                     data['F_L'],
                     data['F_C'],
                     data['L'])
        return model
