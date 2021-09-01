# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.recovery.jakob2019` module.
"""

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (handle_spectral_arguments, reshape_sd,
                                sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_Lab, XYZ_to_xy
from colour.recovery import (XYZ_to_sd_Otsu2018, SPECTRAL_SHAPE_OTSU2018,
                             Dataset_Otsu2018, NodeTree_Otsu2018)
from colour.recovery.otsu2018 import DATASET_REFERENCE_OTSU2018, Data, Node
from colour.utilities import domain_range_scale, metric_mse

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestDataset_Otsu2018', 'TestXYZ_to_sd_Otsu2018', 'TestData', 'TestNode',
    'TestNodeTree_Otsu2018'
]


class TestDataset_Otsu2018(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.Dataset_Otsu2018` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._dataset = DATASET_REFERENCE_OTSU2018
        self._xy = np.array([0.54369557, 0.32107944])

        self._temporary_directory = tempfile.mkdtemp()

        self._path = os.path.join(self._temporary_directory,
                                  'Test_Otsu2018.npz')
        self._dataset.write(self._path)

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('shape', 'basis_functions', 'means',
                               'selector_array')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Dataset_Otsu2018))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', 'select', 'cluster', 'read',
                            'write')

        for method in required_methods:
            self.assertIn(method, dir(Dataset_Otsu2018))

    def test_shape(self):
        """
        Tests :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.shape` property.
        """

        self.assertEqual(self._dataset.shape, SPECTRAL_SHAPE_OTSU2018)

    def test_basis_functions(self):
        """
        Tests :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.basis_functions`
        property.
        """

        self.assertTupleEqual(self._dataset.basis_functions.shape, (8, 3, 36))

    def test_means(self):
        """
        Tests :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.means`
        property.
        """

        self.assertTupleEqual(self._dataset.means.shape, (8, 36))

    def test_selector_array(self):
        """
        Tests :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.selector_array`
        property.
        """

        self.assertTupleEqual(self._dataset.selector_array.shape, (7, 4))

    def test__str__(self):
        """
        Tests :func:`colour.recovery.otsu2018.Dataset_Otsu2018.__str__` method.
        """

        self.assertEqual(
            str(self._dataset), 'Dataset_Otsu2018(8 basis functions)')

    def test_select(self):
        """
        Tests :func:`colour.recovery.otsu2018.Dataset_Otsu2018.select` method.
        """

        self.assertEqual(self._dataset.select(self._xy), 6)

    def test_cluster(self):
        """
        Tests :func:`colour.recovery.otsu2018.Dataset_Otsu2018.cluster` method.
        """

        basis_functions, means = self._dataset.cluster(self._xy)
        self.assertTupleEqual(basis_functions.shape, (3, 36))
        self.assertTupleEqual(means.shape, (36, ))

    def test_read(self):
        """
        Tests :func:`colour.recovery.otsu2018.Dataset_Otsu2018.read` method.
        """

        dataset = Dataset_Otsu2018()
        dataset.read(self._path)

        self.assertEqual(dataset.shape, SPECTRAL_SHAPE_OTSU2018)
        self.assertTupleEqual(dataset.basis_functions.shape, (8, 3, 36))
        self.assertTupleEqual(dataset.means.shape, (8, 36))
        self.assertTupleEqual(dataset.selector_array.shape, (7, 4))

    def test_write(self):
        """
        Tests :func:`colour.recovery.otsu2018.Dataset_Otsu2018.write` method.
        """

        self._dataset.write(self._path)

        dataset = Dataset_Otsu2018()
        dataset.read(self._path)

        self.assertEqual(dataset.shape, SPECTRAL_SHAPE_OTSU2018)
        self.assertTupleEqual(dataset.basis_functions.shape, (8, 3, 36))
        self.assertTupleEqual(dataset.means.shape, (8, 36))
        self.assertTupleEqual(dataset.selector_array.shape, (7, 4))


class TestXYZ_to_sd_Otsu2018(unittest.TestCase):
    """
    Defines :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(
            shape_default=self._shape)
        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

    def test_XYZ_to_sd_Otsu2018(self):
        """
        Tests :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition.
        """

        # Tests the round-trip with values of a colour checker.
        for _name, sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].items():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(
                XYZ, self._cmfs, self._sd_D65, clip=False)
            recovered_XYZ = sd_to_XYZ(recovered_sd, self._cmfs,
                                      self._sd_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = metric_mse(
                reshape_sd(sd, SPECTRAL_SHAPE_OTSU2018).values,
                recovered_sd.values)
            self.assertLess(error, 0.02)

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            self.assertLess(delta_E, 1e-12)

    def test_domain_range_scale_XYZ_to_sd_Otsu2018(self):
        """
        Tests :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_o = sd_to_XYZ(
            XYZ_to_sd_Otsu2018(XYZ_i, self._cmfs, self._sd_D65), self._cmfs,
            self._sd_D65)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ(
                        XYZ_to_sd_Otsu2018(XYZ_i * factor_a, self._cmfs,
                                           self._sd_D65), self._cmfs,
                        self._sd_D65),
                    XYZ_o * factor_b,
                    decimal=7)


class TestData(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.Data` definition unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('tree', 'reflectances', 'reflectances',
                               'basis_functions', 'mean')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Data))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', '__len__', 'PCA',
                            'reconstruct', 'reconstruction_error', 'origin',
                            'partition')

        for method in required_methods:
            self.assertIn(method, dir(Data))


class TestNode(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.Node` definition unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('id', 'tree', 'data', 'children', 'leaves',
                               'partition_axis')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Node))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', '__len__', 'is_leaf',
                            'split', 'find_best_partition',
                            'leaf_reconstruction_error',
                            'branch_reconstruction_error')

        for method in required_methods:
            self.assertIn(method, dir(Node))


class TestNodeTree_Otsu2018(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.NodeTree_Otsu2018` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(
            shape_default=self._shape)
        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('reflectances', 'cmfs', 'illuminant',
                               'minimum_cluster_size')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(NodeTree_Otsu2018))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', 'optimise', 'to_dataset')

        for method in required_methods:
            self.assertIn(method, dir(NodeTree_Otsu2018))

    def test_NodeTree_Otsu2018_and_Dataset_Otsu2018(self):
        """
        Tests :class:`colour.recovery.otsu2018.NodeTree_Otsu2018` dataset
        generation and :class:`colour.recovery.otsu2018.Dataset_Otsu2018`
        input and output. The generated dataset is also tested for
        reconstruction errors.
        """

        reflectances = []
        for colourchecker in ['ColorChecker N Ohta', 'BabelColor Average']:
            for sd in SDS_COLOURCHECKERS[colourchecker].values():
                reflectances.append(reshape_sd(sd, self._shape).values)

        node_tree = NodeTree_Otsu2018(reflectances, self._cmfs, self._sd_D65)
        node_tree.optimise(iterations=5)

        path = os.path.join(self._temporary_directory, 'Test_Otsu2018.npz')
        dataset = node_tree.to_dataset()
        dataset.write(path)

        dataset = Dataset_Otsu2018()
        dataset.read(path)

        for sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].values():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(XYZ, self._cmfs, self._sd_D65,
                                              dataset, False)
            recovered_XYZ = sd_to_XYZ(recovered_sd, self._cmfs,
                                      self._sd_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = metric_mse(
                reshape_sd(sd, SPECTRAL_SHAPE_OTSU2018).values,
                recovered_sd.values)
            self.assertLess(error, 0.075)

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            self.assertLess(delta_E, 1e-12)


if __name__ == '__main__':
    unittest.main()
