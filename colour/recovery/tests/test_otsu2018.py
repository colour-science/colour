"""Define the unit tests for the :mod:`colour.recovery.jakob2019` module."""

import os
import platform
import shutil
import tempfile

import numpy as np
import pytest

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (
    handle_spectral_arguments,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ,
    sds_and_msds_to_msds,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_Lab, XYZ_to_xy
from colour.recovery import (
    SPECTRAL_SHAPE_OTSU2018,
    Dataset_Otsu2018,
    Tree_Otsu2018,
    XYZ_to_sd_Otsu2018,
)
from colour.recovery.otsu2018 import (
    DATASET_REFERENCE_OTSU2018,
    Data_Otsu2018,
    Node_Otsu2018,
    PartitionAxis,
)
from colour.utilities import domain_range_scale, metric_mse

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDataset_Otsu2018",
    "TestXYZ_to_sd_Otsu2018",
    "TestData_Otsu2018",
    "TestNode_Otsu2018",
    "TestTree_Otsu2018",
]


class TestDataset_Otsu2018:
    """
    Define :class:`colour.recovery.otsu2018.Dataset_Otsu2018` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._dataset = DATASET_REFERENCE_OTSU2018
        self._xy = np.array([0.54369557, 0.32107944])

        self._temporary_directory = tempfile.mkdtemp()

        self._path = os.path.join(self._temporary_directory, "Test_Otsu2018.npz")
        self._dataset.write(self._path)

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "shape",
            "basis_functions",
            "means",
            "selector_array",
        )

        for attribute in required_attributes:
            assert attribute in dir(Dataset_Otsu2018)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "select",
            "cluster",
            "read",
            "write",
        )

        for method in required_methods:
            assert method in dir(Dataset_Otsu2018)

    def test_shape(self):
        """Test :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.shape` property."""

        assert self._dataset.shape == SPECTRAL_SHAPE_OTSU2018

    def test_basis_functions(self):
        """
        Test :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.basis_functions`
        property.
        """

        assert self._dataset.basis_functions.shape == (8, 3, 36)

    def test_means(self):
        """
        Test :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.means`
        property.
        """

        assert self._dataset.means.shape == (8, 36)

    def test_selector_array(self):
        """
        Test :attr:`colour.recovery.otsu2018.Dataset_Otsu2018.selector_array`
        property.
        """

        assert self._dataset.selector_array.shape == (7, 4)

    def test__str__(self):
        """Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.__str__` method."""

        assert str(self._dataset) == "Dataset_Otsu2018(8 basis functions)"

        assert str(Dataset_Otsu2018()) == "Dataset_Otsu2018()"

    def test_select(self):
        """Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.select` method."""

        assert self._dataset.select(self._xy) == 6

    def test_raise_exception_select(self):
        """
        Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.select` method
        raised exception.
        """

        pytest.raises(ValueError, Dataset_Otsu2018().select, np.array([0, 0]))

    def test_cluster(self):
        """Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.cluster` method."""

        basis_functions, means = self._dataset.cluster(self._xy)
        assert basis_functions.shape == (3, 36)
        assert means.shape == (36,)

    def test_raise_exception_cluster(self):
        """
        Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.cluster` method
        raised exception.
        """

        pytest.raises(ValueError, Dataset_Otsu2018().cluster, np.array([0, 0]))

    def test_read(self):
        """Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.read` method."""

        dataset = Dataset_Otsu2018()
        dataset.read(self._path)

        assert dataset.shape == SPECTRAL_SHAPE_OTSU2018
        assert dataset.basis_functions.shape == (8, 3, 36)
        assert dataset.means.shape == (8, 36)
        assert dataset.selector_array.shape == (7, 4)

    def test_write(self):
        """Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.write` method."""

        self._dataset.write(self._path)

        dataset = Dataset_Otsu2018()
        dataset.read(self._path)

        assert dataset.shape == SPECTRAL_SHAPE_OTSU2018
        assert dataset.basis_functions.shape == (8, 3, 36)
        assert dataset.means.shape == (8, 36)
        assert dataset.selector_array.shape == (7, 4)

    def test_raise_exception_write(self):
        """
        Test :meth:`colour.recovery.otsu2018.Dataset_Otsu2018.write` method
        raised exception.
        """

        pytest.raises(ValueError, Dataset_Otsu2018().write, "")


class TestXYZ_to_sd_Otsu2018:
    """
    Define :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(shape_default=self._shape)
        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

    def test_XYZ_to_sd_Otsu2018(self):
        """Test :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition."""

        # Tests the round-trip with values of a colour checker.
        for _name, sd in SDS_COLOURCHECKERS["ColorChecker N Ohta"].items():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(XYZ, self._cmfs, self._sd_D65, clip=False)
            recovered_XYZ = sd_to_XYZ(recovered_sd, self._cmfs, self._sd_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = metric_mse(
                reshape_sd(sd, SPECTRAL_SHAPE_OTSU2018).values,
                recovered_sd.values,
            )
            assert error < 0.02

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            assert delta_E < 1e-12

    def test_raise_exception_XYZ_to_sd_Otsu2018(self):
        """
        Test :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition
        raised_exception.
        """

        pytest.raises(
            ValueError,
            XYZ_to_sd_Otsu2018,
            np.array([0, 0, 0]),
            self._cmfs,
            self._sd_D65,
            Dataset_Otsu2018(),
        )

    def test_domain_range_scale_XYZ_to_sd_Otsu2018(self):
        """
        Test :func:`colour.recovery.otsu2018.XYZ_to_sd_Otsu2018` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_o = sd_to_XYZ(
            XYZ_to_sd_Otsu2018(XYZ_i, self._cmfs, self._sd_D65),
            self._cmfs,
            self._sd_D65,
        )

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    sd_to_XYZ(
                        XYZ_to_sd_Otsu2018(XYZ_i * factor_a, self._cmfs, self._sd_D65),
                        self._cmfs,
                        self._sd_D65,
                    ),
                    XYZ_o * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )


class TestData_Otsu2018:
    """
    Define :class:`colour.recovery.otsu2018.Data_Otsu2018` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(shape_default=self._shape)

        self._reflectances = np.transpose(
            reshape_msds(
                sds_and_msds_to_msds(
                    SDS_COLOURCHECKERS["ColorChecker N Ohta"].values()
                ),
                self._shape,
            ).values
        )

        self._data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "reflectances",
            "cmfs",
            "illuminant",
            "basis_functions",
            "mean",
        )

        for attribute in required_attributes:
            assert attribute in dir(Data_Otsu2018)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "__len__",
            "origin",
            "partition",
            "PCA",
            "reconstruct",
            "reconstruction_error",
        )

        for method in required_methods:
            assert method in dir(Data_Otsu2018)

    def test_reflectances(self):
        """
        Test :attr:`colour.recovery.otsu2018.Data_Otsu2018.reflectances`
        property.
        """

        assert self._data.reflectances is self._reflectances

    def test_cmfs(self):
        """Test :attr:`colour.recovery.otsu2018.Data_Otsu2018.cmfs` property."""

        assert self._data.cmfs is self._cmfs

    def test_illuminant(self):
        """
        Test :attr:`colour.recovery.otsu2018.Data_Otsu2018.illuminant`
        property.
        """

        assert self._data.illuminant is self._sd_D65

    def test_basis_functions(self):
        """
        Test :attr:`colour.recovery.otsu2018.Data_Otsu2018.basis_functions`
        property.
        """

        data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        assert data.basis_functions is None

        data.PCA()

        assert data.basis_functions.shape == (3, 36)

    def test_mean(self):
        """Test :attr:`colour.recovery.otsu2018.Data_Otsu2018.mean` property."""

        data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        assert data.mean is None

        data.PCA()

        assert data.mean.shape == (36,)

    def test__str__(self):
        """Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.__str__` method."""

        assert str(self._data) == "Data_Otsu2018(24 Reflectances)"

    def test__len__(self):
        """Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.__len__` method."""

        assert len(self._data) == 24

    def test_origin(self):
        """Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.origin` method."""

        np.testing.assert_allclose(
            self._data.origin(4, 1),
            0.255284008578559,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_raise_exception_origin(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.origin` method
        raised exception.
        """

        pytest.raises(
            ValueError,
            Data_Otsu2018(None, self._cmfs, self._sd_D65).origin,
            4,
            1,
        )

    def test_partition(self):
        """Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.partition` method."""

        partition = self._data.partition(PartitionAxis(4, 1))

        assert len(partition) == 2

    def test_raise_exception_partition(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.partition` method
        raised exception.
        """

        pytest.raises(
            ValueError,
            Data_Otsu2018(None, self._cmfs, self._sd_D65).partition,
            PartitionAxis(4, 1),
        )

    def test_PCA(self):
        """Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.PCA` method."""

        if platform.system() in ("Windows", "Microsoft", "Linux"):
            return

        data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        data.PCA()

        np.testing.assert_allclose(
            data.basis_functions,
            np.array(
                [
                    [
                        0.04391241,
                        0.08560996,
                        0.15556120,
                        0.20826672,
                        0.22981218,
                        0.23117641,
                        0.22718022,
                        0.21742869,
                        0.19854261,
                        0.16868383,
                        0.12020268,
                        0.05958463,
                        -0.01015508,
                        -0.08775193,
                        -0.16957532,
                        -0.23186776,
                        -0.26516404,
                        -0.27409402,
                        -0.27856619,
                        -0.27685075,
                        -0.25597708,
                        -0.21331000,
                        -0.15372029,
                        -0.08746878,
                        -0.02744494,
                        0.01725581,
                        0.04756055,
                        0.07184639,
                        0.09090063,
                        0.10317253,
                        0.10830387,
                        0.10872694,
                        0.10645999,
                        0.10766424,
                        0.11170078,
                        0.11620896,
                    ],
                    [
                        0.03137588,
                        0.06204234,
                        0.11364884,
                        0.17579436,
                        0.20914074,
                        0.22152351,
                        0.23120105,
                        0.24039823,
                        0.24730359,
                        0.25195045,
                        0.25237533,
                        0.24672212,
                        0.23538236,
                        0.22094141,
                        0.20389065,
                        0.18356599,
                        0.15952882,
                        0.13567812,
                        0.11401807,
                        0.09178015,
                        0.06539517,
                        0.03173809,
                        -0.00658524,
                        -0.04710763,
                        -0.08379987,
                        -0.11074555,
                        -0.12606191,
                        -0.13630094,
                        -0.13988107,
                        -0.14193361,
                        -0.14671866,
                        -0.15164795,
                        -0.15772737,
                        -0.16328073,
                        -0.16588768,
                        -0.16947164,
                    ],
                    [
                        -0.01360289,
                        -0.02375832,
                        -0.04262545,
                        -0.07345243,
                        -0.09081235,
                        -0.09227928,
                        -0.08922710,
                        -0.08626299,
                        -0.08584571,
                        -0.08843734,
                        -0.09475094,
                        -0.10376740,
                        -0.11331399,
                        -0.12109706,
                        -0.12678070,
                        -0.13401030,
                        -0.14417036,
                        -0.15408359,
                        -0.16265529,
                        -0.17079814,
                        -0.17972656,
                        -0.19005983,
                        -0.20053986,
                        -0.21017531,
                        -0.21808806,
                        -0.22347400,
                        -0.22650876,
                        -0.22895376,
                        -0.22982598,
                        -0.23001787,
                        -0.23036398,
                        -0.22917409,
                        -0.22684271,
                        -0.22387883,
                        -0.22065773,
                        -0.21821049,
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.mean,
            np.array(
                [
                    0.08795833,
                    0.12050000,
                    0.16787500,
                    0.20675000,
                    0.22329167,
                    0.22837500,
                    0.23229167,
                    0.23579167,
                    0.23658333,
                    0.23779167,
                    0.23866667,
                    0.23975000,
                    0.24345833,
                    0.25054167,
                    0.25791667,
                    0.26150000,
                    0.26437500,
                    0.26566667,
                    0.26475000,
                    0.26554167,
                    0.27137500,
                    0.28279167,
                    0.29529167,
                    0.31070833,
                    0.32575000,
                    0.33829167,
                    0.34675000,
                    0.35554167,
                    0.36295833,
                    0.37004167,
                    0.37854167,
                    0.38675000,
                    0.39587500,
                    0.40266667,
                    0.40683333,
                    0.41287500,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_reconstruct(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.reconstruct`
        method.
        """

        data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        data.PCA()

        np.testing.assert_allclose(
            data.reconstruct(
                np.array(
                    [
                        0.20654008,
                        0.12197225,
                        0.05136952,
                    ]
                )
            ).values,
            np.array(
                [
                    0.06899964,
                    0.08241919,
                    0.09768650,
                    0.08938555,
                    0.07872582,
                    0.07140930,
                    0.06385099,
                    0.05471747,
                    0.04281364,
                    0.03073280,
                    0.01761134,
                    0.00772535,
                    0.00379120,
                    0.00405617,
                    0.00595014,
                    0.01323536,
                    0.03229711,
                    0.05661531,
                    0.07763041,
                    0.10271461,
                    0.14276781,
                    0.20239859,
                    0.27288559,
                    0.35044541,
                    0.42170481,
                    0.47567859,
                    0.50910276,
                    0.53578140,
                    0.55251101,
                    0.56530032,
                    0.58029915,
                    0.59367723,
                    0.60830542,
                    0.62100871,
                    0.62881635,
                    0.63971254,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_raise_exception_reconstruct(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.reconstruct` method
        raised exception.
        """

        pytest.raises(
            ValueError,
            Data_Otsu2018(None, self._cmfs, self._sd_D65).reconstruct,
            np.array([0, 0, 0]),
        )

    def test_reconstruction_error(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.\
reconstruction_error` method.
        """

        data = Data_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        np.testing.assert_allclose(
            data.reconstruction_error(),
            2.753352549148681,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_raise_exception_reconstruction_error(self):
        """
        Test :meth:`colour.recovery.otsu2018.Data_Otsu2018.\
reconstruction_error` method raised exception.
        """

        pytest.raises(
            ValueError,
            Data_Otsu2018(None, self._cmfs, self._sd_D65).reconstruction_error,
        )


class TestNode_Otsu2018:
    """
    Define :class:`colour.recovery.otsu2018.Node_Otsu2018` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(shape_default=self._shape)

        self._reflectances = sds_and_msds_to_msds(
            SDS_COLOURCHECKERS["ColorChecker N Ohta"].values()
        )

        self._tree = Tree_Otsu2018(self._reflectances)
        self._tree.optimise()
        for leaf in self._tree.leaves:
            if len(leaf.parent.children) == 2:
                self._node_a = leaf.parent
                self._node_b, self._node_c = self._node_a.children
                break

        self._data_a = Data_Otsu2018(
            np.transpose(reshape_msds(self._reflectances, self._shape).values),
            self._cmfs,
            self._sd_D65,
        )
        self._data_b = self._node_b.data

        self._partition_axis = self._node_a.partition_axis

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("partition_axis", "row")

        for attribute in required_attributes:
            assert attribute in dir(Node_Otsu2018)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "split",
            "minimise",
            "leaf_reconstruction_error",
            "branch_reconstruction_error",
        )

        for method in required_methods:
            assert method in dir(Node_Otsu2018)

    def test_partition_axis(self):
        """
        Test :attr:`colour.recovery.otsu2018.Node_Otsu2018.partition_axis`
        property.
        """

        assert self._node_a.partition_axis is self._partition_axis

    def test_row(self):
        """Test :attr:`colour.recovery.otsu2018.Node_Otsu2018.row` property."""

        assert self._node_a.row == (
            self._partition_axis.origin,
            self._partition_axis.direction,
            self._node_b,
            self._node_c,
        )

    def test_raise_exception_row(self):
        """
        Test :attr:`colour.recovery.otsu2018.Node_Otsu2018.row` property
        raised exception.
        """

        pytest.raises(ValueError, lambda: Node_Otsu2018().row)

    def test_split(self):
        """Test :meth:`colour.recovery.otsu2018.Node_Otsu2018.split` method."""

        node_a = Node_Otsu2018(self._tree, None)
        node_b = Node_Otsu2018(self._tree, data=self._data_a)
        node_c = Node_Otsu2018(self._tree, data=self._data_a)
        node_a.split([node_b, node_c], PartitionAxis(12, 0))

        assert len(node_a.children) == 2

    def test_minimise(self):
        """Test :meth:`colour.recovery.otsu2018.Node_Otsu2018.minimise` method."""

        node = Node_Otsu2018(data=self._data_a)
        partition, axis, partition_error = node.minimise(3)

        assert (len(partition[0].data), len(partition[1].data)) == (10, 14)

        np.testing.assert_allclose(
            axis.origin, 0.324111380117147, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            partition_error, 2.0402980027, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_leaf_reconstruction_error(self):
        """
        Test :meth:`colour.recovery.otsu2018.Node_Otsu2018.\
leaf_reconstruction_error` method.
        """

        np.testing.assert_allclose(
            self._node_b.leaf_reconstruction_error(),
            1.145340908277367e-29,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_branch_reconstruction_error(self):
        """
        Test :meth:`colour.recovery.otsu2018.Node_Otsu2018.\
branch_reconstruction_error` method.
        """

        np.testing.assert_allclose(
            self._node_a.branch_reconstruction_error(),
            3.900015991807948e-25,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestTree_Otsu2018:
    """
    Define :class:`colour.recovery.otsu2018.Tree_Otsu2018` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_OTSU2018
        self._cmfs, self._sd_D65 = handle_spectral_arguments(shape_default=self._shape)

        self._reflectances = sds_and_msds_to_msds(
            list(SDS_COLOURCHECKERS["ColorChecker N Ohta"].values())
            + list(SDS_COLOURCHECKERS["BabelColor Average"].values())
        )

        self._tree = Tree_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)

        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

        self._temporary_directory = tempfile.mkdtemp()

        self._path = os.path.join(self._temporary_directory, "Test_Otsu2018.npz")

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("reflectances", "cmfs", "illuminant")

        for attribute in required_attributes:
            assert attribute in dir(Tree_Otsu2018)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__", "__str__", "optimise", "to_dataset")

        for method in required_methods:
            assert method in dir(Tree_Otsu2018)

    def test_reflectances(self):
        """
        Test :attr:`colour.recovery.otsu2018.Tree_Otsu2018.reflectances`
        property.
        """

        np.testing.assert_allclose(
            self._tree.reflectances,
            np.transpose(
                reshape_msds(
                    sds_and_msds_to_msds(self._reflectances), self._shape
                ).values
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_cmfs(self):
        """Test :attr:`colour.recovery.otsu2018.Tree_Otsu2018.cmfs` property."""

        assert self._tree.cmfs is self._cmfs

    def test_illuminant(self):
        """
        Test :attr:`colour.recovery.otsu2018.Tree_Otsu2018.illuminant`
        property.
        """

        assert self._tree.illuminant is self._sd_D65

    def test_optimise(self):
        """Test :class:`colour.recovery.otsu2018.Tree_Otsu2018.optimise` method."""

        node_tree = Tree_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)
        node_tree.optimise(iterations=5)

        dataset = node_tree.to_dataset()
        dataset.write(self._path)

        dataset = Dataset_Otsu2018()
        dataset.read(self._path)

        for sd in SDS_COLOURCHECKERS["ColorChecker N Ohta"].values():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = XYZ_to_sd_Otsu2018(
                XYZ, self._cmfs, self._sd_D65, dataset, False
            )
            recovered_XYZ = sd_to_XYZ(recovered_sd, self._cmfs, self._sd_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = metric_mse(
                reshape_sd(sd, SPECTRAL_SHAPE_OTSU2018).values,
                recovered_sd.values,
            )
            assert error < 0.075

            delta_E = delta_E_CIE1976(Lab, recovered_Lab)
            assert delta_E < 1e-12

    def test_to_dataset(self):
        """
        Test :attr:`colour.recovery.otsu2018.Tree_Otsu2018.to_dataset`
        method.
        """

        node_tree = Tree_Otsu2018(self._reflectances, self._cmfs, self._sd_D65)
        dataset = node_tree.to_dataset()
        dataset.write(self._path)
