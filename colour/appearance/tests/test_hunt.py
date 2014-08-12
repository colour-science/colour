from colour.appearance.hunt import XYZ_to_Hunt
from colour.appearance.tests.common import ColorAppearanceTest


class TestHuntColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'hunt.csv'

    output_parameter_dict = {'h_S': 'h_S',
                             's': 's',
                             'Q': 'Q',
                             'J': 'J',
                             'C_94': 'C94',
                             'M94': 'M94'}

    def create_model_from_data(self, data):
        model = XYZ_to_Hunt(data['X'], data['Y'], data['Z'],
                     data['X_W'], 0.2 * data['Y_W'], data['Z_W'],
                     data['X_W'], data['Y_W'], data['Z_W'],
                     l_a=data['L_A'],
                     n_c=data['N_c'],
                     n_b=data['N_b'],
                     cct_w=data['T'])

        return model