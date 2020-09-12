# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.jakob2019` module.
"""

from __future__ import division, unicode_literals

import unittest
import os
import shutil
import tempfile

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (CCS_ILLUMINANTS, SDS_ILLUMINANTS,
                                MSDS_CMFS_STANDARD_OBSERVER,
                                SpectralDistribution, sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_Lab
from colour.recovery import (XYZ_to_sd_Otsu2018, SPECTRAL_SHAPE_OTSU2018,
                             Dataset_Otsu2018, NodeTree_Otsu2018)
from colour.recovery.otsu2018 import ColourData, Node
from colour.utilities import as_float_array, metric_mse

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestDataset_Otsu2018', 'TestXYZ_to_sd_Otsu2018', 'TestColourData',
    'TestNode', 'TestNodeTree_Otsu2018'
]

CMFS = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
SD_D65 = SpectralDistribution(SDS_ILLUMINANTS['D65'])
CCS_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']


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

        required_methods = ('write', 'read', 'select', 'cluster')

        for method in required_methods:
            self.assertIn(method, dir(Dataset_Otsu2018))


class TestXYZ_to_sd_Otsu2018(unittest.TestCase):
    """
    Defines :func:`colour.recovery.otsu2018.XYZ_to_sd_otsu2018` definition unit
    tests methods.
    """

    def test_XYZ_to_sd_otsu2018(self):
        """
        Tests :func:`colour.recovery.otsu2018.XYZ_to_sd_otsu2018` definition.
        """

        # Tests the round-trip with values of a colour checker.
        for _name, sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].items():
            XYZ = sd_to_XYZ(sd, illuminant=SD_D65) / 100
            Lab = XYZ_to_Lab(XYZ, CCS_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(XYZ, CMFS, SD_D65, clip=False)
            recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=SD_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, CCS_D65)

            error = metric_mse(sd.copy().align(SPECTRAL_SHAPE_OTSU2018).values,
                               recovered_sd.values)
            self.assertLess(error, 0.02)

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            self.assertLess(delta_E, 1e-12)


class TestColourData(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.ColourData` definition unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('tree', 'reflectances', 'XYZ', 'xy')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(ColourData))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__len__', 'partition')

        for method in required_methods:
            self.assertIn(method, dir(ColourData))


class TestNode(unittest.TestCase):
    """
    Defines :class:`colour.recovery.otsu2018.Node` definition unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('id', 'tree', 'colour_data', 'children',
                               'partition_axis', 'basis_functions', 'mean',
                               'leaves')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Node))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__len__', 'is_leaf', 'split', 'PCA',
                            'reconstruct', 'leaf_reconstruction_error',
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

        required_attributes = ('reflectances', 'shape', 'cmfs', 'illuminant',
                               'minimum_cluster_size')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(NodeTree_Otsu2018))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', 'msds_to_XYZ', 'optimise', 'to_dataset')

        for method in required_methods:
            self.assertIn(method, dir(NodeTree_Otsu2018))

    def test_NodeTree_Otsu2018_and_Dataset_Otsu2018(self):
        """
        Tests :class:`colour.recovery.otsu2018.NodeTree_Otsu2018` dataset
        generation and :class:`colour.recovery.otsu2018.Dataset_Otsu2018`
        input and output. The generated dataset is also tested for
        reconstruction errors.
        """

        data = []
        for colourchecker in ['ColorChecker N Ohta', 'BabelColor Average']:
            for sd in SDS_COLOURCHECKERS[colourchecker].values():
                data.append(sd.copy().align(SPECTRAL_SHAPE_OTSU2018).values)

        data = as_float_array(data)
        tree = NodeTree_Otsu2018(data, SPECTRAL_SHAPE_OTSU2018)
        tree.optimise(iterations=2)

        path = os.path.join(self._temporary_directory, 'Otsu2018_Test.npz')
        dataset = tree.to_dataset()
        dataset.write(path)

        dataset = Dataset_Otsu2018()
        dataset.read(path)

        for sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].values():
            XYZ = sd_to_XYZ(sd, illuminant=SD_D65) / 100
            Lab = XYZ_to_Lab(XYZ, CCS_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(
                XYZ, CMFS, SD_D65, clip=False, dataset=dataset)
            recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=SD_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, CCS_D65)

            error = metric_mse(sd.copy().align(SPECTRAL_SHAPE_OTSU2018).values,
                               recovered_sd.values)
            self.assertLess(error, 0.075)

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            self.assertLess(delta_E, 1e-12)


if __name__ == '__main__':
    unittest.main()
