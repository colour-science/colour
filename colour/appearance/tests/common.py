"""
Copyright (c) 2014, Michael Mauderer, University of St Andrews
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of St Andrews nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from abc import abstractmethod
from collections import defaultdict
import csv
import os

import unittest
import numpy

from numpy.testing import assert_allclose, assert_almost_equal


class ColorAppearanceTest():
    fixture_path = None

    @staticmethod
    def load_fixture(file_name):
        print(file_name)
        path = os.path.dirname(__file__)
        with open(os.path.join(path, 'fixtures', file_name)) as in_file:
            result = []
            for case_data in csv.DictReader(in_file):
                for key in case_data:
                    try:
                        case_data[key] = float(case_data[key])
                    except ValueError:
                        pass
                result.append(case_data)
            return result

    def check_model_consistency(self, data, output_parameter_dict):
        for data_attr, model_attr in sorted(output_parameter_dict.items()):
            yield self.check_model_attribute, data.get('Case'), data, model_attr, data[data_attr]

    @abstractmethod
    def create_model_from_data(self, data):
        pass

    def check_model_attribute(self, case, data, model_attr, target):
        model = self.create_model_from_data(data)
        model_parameter = getattr(model, model_attr)
        error_message = 'Parameter {} in test case {} does not match target value.\nExpected: {} \nReceived {}'.format(
            model_attr, case, target, model_parameter)

        assert_allclose(model_parameter, target, err_msg=error_message, rtol=0.01, atol=0.01, verbose=False)
        assert_almost_equal(model_parameter, target, decimal=1, err_msg=error_message)

    limited_fixtures = None

    def _get_fixtures(self):
        # Sometimes it might be desirable to exclude s specific fixture for testing
        fixtures = self.load_fixture(self.fixture_path)
        if self.limited_fixtures is not None:
            fixtures = [fixtures[index] for index in self.limited_fixtures]
        return fixtures

    def test_forward_examples(self):
        # Go through all available fixtures
        for data in self._get_fixtures():
            # Create a single test for each output parameter
            for test in self.check_model_consistency(data, self.output_parameter_dict):
                yield test

    @unittest.skip
    def test_parallel_forward_example(self):
        # Collect all fixture data in a single dict of lists
        data = defaultdict(list)
        for fixture in self._get_fixtures():
            for key, value in fixture.items():
                data[key].append(value)
        # Turn lists into numpy.arrays
        for key in data:
            data[key] = numpy.array(data[key])
        # Create tests
        for test in self.check_model_consistency(data, self.output_parameter_dict):
            yield test


