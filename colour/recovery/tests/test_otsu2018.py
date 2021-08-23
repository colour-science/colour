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
from colour.colorimetry import (MSDS_CMFS_STANDARD_OBSERVER, CCS_ILLUMINANTS,
                                SDS_ILLUMINANTS, reshape_msds, reshape_sd,
                                sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_Lab
from colour.recovery import (XYZ_to_sd_Otsu2018, SPECTRAL_SHAPE_OTSU2018,
                             Dataset_Otsu2018, NodeTree_Otsu2018)
from colour.recovery.otsu2018 import Data, Node
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

_MSDS_CMFS_DEFAULT = 'CIE 1931 2 Degree Standard Observer'

_ILLUMINANT_DEFAULT = 'D65'


class TestDataset_Otsu2018(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.Dataset_Otsu2018` definition unit
    tests methods.
    """

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

        required_methods = ('__init__', 'select', 'cluster', 'read', 'write')

        for method in required_methods:
            self.assertIn(method, dir(Dataset_Otsu2018))


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
        # pylint: disable=E1102
        self._cmfs = reshape_msds(
            MSDS_CMFS_STANDARD_OBSERVER[_MSDS_CMFS_DEFAULT], self._shape)

        self._sd_D65 = reshape_sd(SDS_ILLUMINANTS[_ILLUMINANT_DEFAULT],
                                  self._shape)
        self._xy_D65 = CCS_ILLUMINANTS[_MSDS_CMFS_DEFAULT][_ILLUMINANT_DEFAULT]

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

        required_attributes = ('tree', 'reflectances', 'XYZ', 'xy')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Data))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', '__len__', 'partition')

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

        required_attributes = ('id', 'tree', 'data', 'children',
                               'partition_axis', 'basis_functions', 'mean',
                               'leaves')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Node))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__str__', '__len__', 'is_leaf',
                            'split', 'PCA', 'reconstruct',
                            'leaf_reconstruction_error',
                            'branch_reconstruction_error',
                            'partition_reconstruction_error',
                            'find_best_partition')

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
        # pylint: disable=E1102
        self._cmfs = reshape_msds(
            MSDS_CMFS_STANDARD_OBSERVER[_MSDS_CMFS_DEFAULT], self._shape)

        self._sd_D65 = reshape_sd(SDS_ILLUMINANTS[_ILLUMINANT_DEFAULT],
                                  self._shape)
        self._xy_D65 = CCS_ILLUMINANTS[_MSDS_CMFS_DEFAULT][_ILLUMINANT_DEFAULT]

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
        node_tree.optimise(iterations=2)

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
