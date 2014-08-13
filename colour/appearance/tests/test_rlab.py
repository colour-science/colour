from colour.appearance.rlab import XYZ_to_RLAB
from colour.appearance.tests.common import ColorAppearanceTest


class TestRLABColorAppearanceModel(ColorAppearanceTest):
    fixture_path = 'rlab.csv'
    output_parameter_dict = {'L': 'L',
                             'C': 'C',
                             's': 's',
                             'a': 'a',
                             'b': 'b',
                             'h': 'h'}

    def create_model_from_data(self, data):
        model = XYZ_to_RLAB(data['X'], data['Y'], data['Z'],
                     data['X_n'], data['Y_n'], data['Z_n'],
                     data['Y_n2'],
                     data['sigma'],
                     data['D'])
        return model

