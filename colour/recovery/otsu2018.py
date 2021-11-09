# -*- coding: utf-8 -*-
"""
Otsu, Yamamoto and Hachisuka (2018) - Reflectance Recovery
==========================================================

Defines the objects for reflectance recovery, i.e. spectral upsampling, using
*Otsu et al. (2018)* method:

-   :class:`colour.recovery.Dataset_Otsu2018`
-   :func:`colour.recovery.XYZ_to_sd_Otsu2018`
-   :func:`colour.recovery.Tree_Otsu2018`

References
----------
-   :cite:`Otsu2018` : Otsu, H., Yamamoto, M., & Hachisuka, T. (2018).
    Reproducing Spectral Reflectances From Tristimulus Colours. Computer
    Graphics Forum, 37(6), 370-381. doi:10.1111/cgf.13332
"""

import numpy as np
from collections import namedtuple

from colour.colorimetry import (
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    msds_to_XYZ_integration,
    reshape_msds,
    sd_to_XYZ,
)
from colour.models import XYZ_to_xy
from colour.recovery import (
    SPECTRAL_SHAPE_OTSU2018,
    BASIS_FUNCTIONS_OTSU2018,
    CLUSTER_MEANS_OTSU2018,
    SELECTOR_ARRAY_OTSU2018,
)
from colour.utilities import (
    Node,
    as_float_array,
    domain_range_scale,
    is_tqdm_installed,
    message_box,
    to_domain_1,
    zeros,
)

if is_tqdm_installed():
    from tqdm import tqdm
else:  # pragma: no cover
    from unittest import mock

    tqdm = mock.MagicMock()

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'Dataset_Otsu2018',
    'DATASET_REFERENCE_OTSU2018',
    'XYZ_to_sd_Otsu2018',
    'PartitionAxis',
    'Data_Otsu2018',
    'Node_Otsu2018',
    'Tree_Otsu2018',
]


class Dataset_Otsu2018:
    """
    Stores all the information needed for the *Otsu et al. (2018)* spectral
    upsampling method.

    Datasets can be either generated and converted as a
    :class:`colour.recovery.Dataset_Otsu2018` class instance using the
    :meth:`colour.recovery.Tree_Otsu2018.to_dataset` method or
    alternatively, loaded from disk with the
    :meth:`colour.recovery.Dataset_Otsu2018.read` method.

    Parameters
    ----------
    shape: SpectralShape
        Shape of the spectral data.
    basis_functions : array_like, (n, 3, m)
        Three basis functions for every cluster.
    means : array_like, (n, m)
        Mean for every cluster.
    selector_array : array_like, (k, 4)
        Array describing how to select the appropriate cluster. See
        :meth:`colour.recovery.Dataset_Otsu2018.select` method for details.

    Attributes
    ----------
    -   :attr:`~colour.recovery.Dataset_Otsu2018.shape`
    -   :attr:`~colour.recovery.Dataset_Otsu2018.basis_functions`
    -   :attr:`~colour.recovery.Dataset_Otsu2018.means`
    -   :attr:`~colour.recovery.Dataset_Otsu2018.selector_array`

    Methods
    -------
    -   :meth:`~colour.recovery.Dataset_Otsu2018.__init__`
    -   :meth:`~colour.recovery.Dataset_Otsu2018.select`
    -   :meth:`~colour.recovery.Dataset_Otsu2018.cluster`
    -   :meth:`~colour.recovery.Dataset_Otsu2018.read`
    -   :meth:`~colour.recovery.Dataset_Otsu2018.write`

    References
    ----------
    :cite:`Otsu2018`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> from colour.characterisation import SDS_COLOURCHECKERS
    >>> from colour.colorimetry import sds_and_msds_to_msds
    >>> reflectances = sds_and_msds_to_msds(
    ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
    ... )
    >>> node_tree = Tree_Otsu2018(reflectances)
    >>> node_tree.optimise(iterations=2, print_callable=lambda x: x)
    >>> dataset = node_tree.to_dataset()
    >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
    ...                     'resources', 'ColorChecker_Otsu2018.npz')
    >>> dataset.write(path) # doctest: +SKIP
    >>> dataset = Dataset_Otsu2018() # doctest: +SKIP
    >>> dataset.read(path) # doctest: +SKIP
    """

    def __init__(self,
                 shape=None,
                 basis_functions=None,
                 means=None,
                 selector_array=None):
        self._shape = shape
        self._basis_functions = as_float_array(basis_functions)
        self._means = as_float_array(means)
        self._selector_array = as_float_array(selector_array)

    @property
    def shape(self):
        """
        Getter property for the shape used by the *Otsu et al. (2018)* dataset.

        Returns
        -------
        SpectralShape
            Shape used by the *Otsu et al. (2018)* dataset.
        """

        return self._shape

    @property
    def basis_functions(self):
        """
        Getter property for the basis functions of the *Otsu et al. (2018)*
        dataset.

        Returns
        -------
        ndarray
            Basis functions of the *Otsu et al. (2018)* dataset.
        """

        return self._basis_functions

    @property
    def means(self):
        """
        Getter property for means of the *Otsu et al. (2018)* dataset.

        Returns
        -------
        int
            Means of the *Otsu et al. (2018)* dataset.
        """

        return self._means

    @property
    def selector_array(self):
        """
        Getter property for the selector array of the *Otsu et al. (2018)*
        dataset.

        Returns
        -------
        ndarray
            Selector array of the *Otsu et al. (2018)* dataset.
        """

        return self._selector_array

    def __str__(self):
        """
        Returns a formatted string representation of the dataset.

        Returns
        -------
        str
            Formatted string representation.
        """

        return '{0}({1} basis functions)'.format(
            self.__class__.__name__, self._basis_functions.shape[0])

    def select(self, xy):
        """
        Returns the cluster index appropriate for the given *CIE xy*
        coordinates.

        Parameters
        ----------
        xy : array_like, (2,)
            *CIE xy* chromaticity coordinates.

        Returns
        -------
        int
            Cluster index.
        """

        i = 0
        while True:
            row = self._selector_array[i, :]
            direction, origin, lesser_index, greater_index = row

            if xy[int(direction)] <= origin:
                index = int(lesser_index)
            else:
                index = int(greater_index)

            if index < 0:
                i = -index
            else:
                return index

    def cluster(self, xy):
        """
        Returns the basis functions and dataset mean for the given *CIE xy*
        coordinates.

        Parameters
        ----------
        xy : array_like, (2,)
            *CIE xy* chromaticity coordinates.

        Returns
        -------
        basis_functions : ndarray, (3, n)
            Three basis functions.
        mean : ndarray, (n,)
            Dataset mean.
        """

        index = self.select(xy)

        return self._basis_functions[index, :, :], self._means[index, :]

    def read(self, path):
        """
        Reads and loads a dataset from an *.npz* file.

        Parameters
        ----------
        path : str
            Path to the file.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour.characterisation import SDS_COLOURCHECKERS
        >>> from colour.colorimetry import sds_and_msds_to_msds
        >>> reflectances = sds_and_msds_to_msds(
        ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
        ... )
        >>> node_tree = Tree_Otsu2018(reflectances)
        >>> node_tree.optimise(iterations=2, print_callable=lambda x: x)
        >>> dataset = node_tree.to_dataset()
        >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
        ...                     'resources', 'ColorChecker_Otsu2018.npz')
        >>> dataset.write(path) # doctest: +SKIP
        >>> dataset = Dataset_Otsu2018() # doctest: +SKIP
        >>> dataset.read(path) # doctest: +SKIP
        """

        data = np.load(path)

        start, end, interval = data['shape']
        self._shape = SpectralShape(start, end, interval)
        self._basis_functions = data['basis_functions']
        self._means = data['means']
        self._selector_array = data['selector_array']

    def write(self, path):
        """
        Writes the dataset to an *.npz* file at given path.

        Parameters
        ----------
        path : str
            Path to the file.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour.characterisation import SDS_COLOURCHECKERS
        >>> from colour.colorimetry import sds_and_msds_to_msds
        >>> reflectances = sds_and_msds_to_msds(
        ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
        ... )
        >>> node_tree = Tree_Otsu2018(reflectances)
        >>> node_tree.optimise(iterations=2, print_callable=lambda x: x)
        >>> dataset = node_tree.to_dataset()
        >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
        ...                     'resources', 'ColorChecker_Otsu2018.npz')
        >>> dataset.write(path) # doctest: +SKIP
        """

        shape_array = as_float_array(
            [self._shape.start, self._shape.end, self._shape.interval])

        np.savez(
            path,
            shape=shape_array,
            basis_functions=self._basis_functions,
            means=self._means,
            selector_array=self._selector_array)


DATASET_REFERENCE_OTSU2018 = Dataset_Otsu2018(
    SPECTRAL_SHAPE_OTSU2018, BASIS_FUNCTIONS_OTSU2018, CLUSTER_MEANS_OTSU2018,
    SELECTOR_ARRAY_OTSU2018)
"""
Builtin *Otsu et al. (2018)* dataset as a
:class:`colour.recovery.Dataset_Otsu2018` class instance, usable by
:func:`colour.recovery.XYZ_to_sd_Otsu2018` definition among others.
"""


def XYZ_to_sd_Otsu2018(XYZ,
                       cmfs=None,
                       illuminant=None,
                       dataset=DATASET_REFERENCE_OTSU2018,
                       clip=True):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus values
    using *Otsu et al. (2018)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    dataset : Dataset_Otsu2018, optional
        Dataset to use for reconstruction. The default is to use the published
        data.
    clip : bool, optional
        If *True*, the default, values below zero and above unity in the
        recovered spectral distributions will be clipped. This ensures that the
        returned reflectance is physical and conserves energy, but will cause
        noticeable colour differences in case of very saturated colours.

    Returns
    -------
    SpectralDistribution
        Recovered spectral distribution. Its shape is always that of the
        :class:`colour.recovery.SPECTRAL_SHAPE_OTSU2018` class instance.

    References
    ----------
    :cite:`Otsu2018`

    Examples
    --------
    >>> from colour import (
    ...     CCS_ILLUMINANTS, SDS_ILLUMINANTS, MSDS_CMFS, XYZ_to_sRGB)
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SPECTRAL_SHAPE_OTSU2018)
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd_Otsu2018(XYZ, cmfs, illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.0601939...],
                          [ 390.        ,    0.0568063...],
                          [ 400.        ,    0.0517429...],
                          [ 410.        ,    0.0495841...],
                          [ 420.        ,    0.0502007...],
                          [ 430.        ,    0.0506489...],
                          [ 440.        ,    0.0510020...],
                          [ 450.        ,    0.0493782...],
                          [ 460.        ,    0.0468046...],
                          [ 470.        ,    0.0437132...],
                          [ 480.        ,    0.0416957...],
                          [ 490.        ,    0.0403783...],
                          [ 500.        ,    0.0405197...],
                          [ 510.        ,    0.0406031...],
                          [ 520.        ,    0.0416912...],
                          [ 530.        ,    0.0430956...],
                          [ 540.        ,    0.0444474...],
                          [ 550.        ,    0.0459336...],
                          [ 560.        ,    0.0507631...],
                          [ 570.        ,    0.0628967...],
                          [ 580.        ,    0.0844661...],
                          [ 590.        ,    0.1334277...],
                          [ 600.        ,    0.2262428...],
                          [ 610.        ,    0.3599330...],
                          [ 620.        ,    0.4885571...],
                          [ 630.        ,    0.5752546...],
                          [ 640.        ,    0.6193023...],
                          [ 650.        ,    0.6450744...],
                          [ 660.        ,    0.6610548...],
                          [ 670.        ,    0.6688673...],
                          [ 680.        ,    0.6795426...],
                          [ 690.        ,    0.6887933...],
                          [ 700.        ,    0.7003469...],
                          [ 710.        ,    0.7084128...],
                          [ 720.        ,    0.7154674...],
                          [ 730.        ,    0.7234334...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2065494...,  0.1219712...,  0.0514002...])
    """

    XYZ = to_domain_1(XYZ)

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, shape_default=SPECTRAL_SHAPE_OTSU2018)

    xy = XYZ_to_xy(XYZ)

    basis_functions, mean = dataset.cluster(xy)

    M = np.empty((3, 3))
    for i in range(3):
        sd = SpectralDistribution(basis_functions[i, :], dataset.shape.range())

        with domain_range_scale('ignore'):
            M[:, i] = sd_to_XYZ(sd, cmfs, illuminant) / 100

    M_inverse = np.linalg.inv(M)

    sd = SpectralDistribution(mean, dataset.shape.range())

    with domain_range_scale('ignore'):
        XYZ_mu = sd_to_XYZ(sd, cmfs, illuminant) / 100

    weights = np.dot(M_inverse, XYZ - XYZ_mu)
    recovered_sd = np.dot(weights, basis_functions) + mean

    recovered_sd = np.clip(recovered_sd, 0, 1) if clip else recovered_sd

    return SpectralDistribution(recovered_sd, dataset.shape.range())


class PartitionAxis(namedtuple('PartitionAxis', ('origin', 'direction'))):
    """
    Represents a horizontal or vertical line, partitioning the 2D space in
    two half-planes.

    Parameters
    ----------
    origin : numeric
        The x coordinate of a vertical line or the y coordinate of a horizontal
        line.
    direction : int
        *0* if vertical, *1* if horizontal.

    Methods
    -------
    -   :meth:`~colour.recovery.otsu2018.PartitionAxis.__str__`
    """

    def __str__(self):
        """
        Returns a formatted string representation of the partition axis.

        Returns
        -------
        str
            Formatted string representation.
        """

        return '{0}({1} partition at {2} = {3})'.format(
            self.__class__.__name__, 'horizontal'
            if self.direction else 'vertical', 'y'
            if self.direction else 'x', self.origin)


class Data_Otsu2018:
    """
    Stores the reference reflectances and derived information along with the
    methods to process them for a leaf :class:`colour.recovery.otsu2018.Node`
    class instance.

    This class also supports partitioning: Creating two smaller instances of
    :class:`colour.recovery.otsu2018.Data` class by splitting along an
    horizontal or a vertical axis on the *CIE xy* plane.

    Parameters
    ----------
    reflectances : ndarray, (n, m), optional
        Reference reflectances of the *n* colours to be stored.
        The shape must match ``tree.shape`` with *m* points for each colour.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.

    Attributes
    ----------
    -   :attr:`~colour.recovery.otsu2018.Data.reflectances`
    -   :attr:`~colour.recovery.otsu2018.Data.cmfs`
    -   :attr:`~colour.recovery.otsu2018.Data.illuminant`
    -   :attr:`~colour.recovery.otsu2018.Data.basis_functions`
    -   :attr:`~colour.recovery.otsu2018.Data.mean`

    Methods
    -------
    -   :meth:`~colour.recovery.otsu2018.Data.__str__`
    -   :meth:`~colour.recovery.otsu2018.Data.__len__`
    -   :meth:`~colour.recovery.otsu2018.Data.origin`
    -   :meth:`~colour.recovery.otsu2018.Data.partition`
    -   :meth:`~colour.recovery.otsu2018.Data.PCA`
    -   :meth:`~colour.recovery.otsu2018.Data.reconstruct`
    -   :meth:`~colour.recovery.otsu2018.Data.reconstruction_error`
    """

    def __init__(self, reflectances, cmfs, illuminant):
        self._cmfs = cmfs
        self._illuminant = illuminant

        self._XYZ = None
        self._xy = None

        self._reflectances = None
        self.reflectances = reflectances

        self._mean = None
        self._basis_functions = None
        self._M = None
        self._XYZ_mu = None

        self._reconstruction_error = None

    @property
    def reflectances(self):
        """
        Getter and setter property for the reference reflectances.

        Parameters
        ----------
        value : array_like
            Value to set the reference reflectances with.

        Returns
        -------
        ndarray
            Reference reflectances.
        """

        return self._reflectances

    @reflectances.setter
    def reflectances(self, value):
        """
        Setter for the **self.reflectances** property.
        """

        if value is not None:
            self._reflectances = as_float_array(value)

            self._XYZ = msds_to_XYZ_integration(
                self._reflectances,
                self._cmfs,
                self._illuminant,
                shape=self._cmfs.shape) / 100

            self._xy = XYZ_to_xy(self._XYZ)

    @property
    def cmfs(self):
        """
        Getter property for the standard observer colour matching functions.

        Returns
        -------
        XYZ_ColourMatchingFunctions
            Standard observer colour matching functions.
        """

        return self._cmfs

    @property
    def illuminant(self):
        """
        Getter property for the illuminant.

        Returns
        -------
        SpectralDistribution
            Illuminant.
        """

        return self._illuminant

    @property
    def basis_functions(self):
        """
        Getter property for the basis functions.

        Returns
        -------
        array_like
            Basis functions.
        """

        return self._basis_functions

    @property
    def mean(self):
        """
        Getter property for the mean distribution.

        Returns
        -------
        array_like
            Mean distribution.
        """

        return self._mean

    def __str__(self):
        """
        Returns a formatted string representation of the data.

        Returns
        -------
        str
            Formatted string representation.
        """

        return '{0}({1} Reflectances)'.format(self.__class__.__name__,
                                              len(self))

    def __len__(self):
        """
        Returns the number of colours in the data.

        Returns
        -------
        int
            Number of colours in the data.
        """

        return self._reflectances.shape[0]

    def origin(self, i, direction):
        """
        Returns the origin *CIE x* or *CIE y* chromaticity coordinate for given
        index and direction.

        Parameters
        ----------
        i : int
            Origin index.
        direction : int
            Origin direction.

        Returns
        -------
        float
            Origin *CIE x* or *CIE y* chromaticity coordinate.
        """

        return self._xy[i, direction]

    def partition(self, axis):
        """
        Partitions the data using given partition axis.

        Parameters
        ----------
        axis : PartitionAxis
            Partition axis used to partition the data.

        Returns
        -------
        lesser : Data_Otsu2018
            The left or lower part.
        greater : Data_Otsu2018
            The right or upper part.
        """

        lesser = Data_Otsu2018(None, self._cmfs, self._illuminant)
        greater = Data_Otsu2018(None, self._cmfs, self._illuminant)

        mask = self._xy[:, axis.direction] <= axis.origin

        lesser._reflectances = self._reflectances[mask, :]
        greater._reflectances = self._reflectances[~mask, :]

        lesser._XYZ = self._XYZ[mask, :]
        greater._XYZ = self._XYZ[~mask, :]

        lesser._xy = self._xy[mask, :]
        greater._xy = self._xy[~mask, :]

        return lesser, greater

    def PCA(self):
        """
        Performs the *Principal Component Analysis* (PCA) on the data and sets
        the relevant attributes accordingly.
        """

        if self._M is not None:
            return

        settings = {
            'cmfs': self._cmfs,
            'illuminant': self._illuminant,
            'shape': self._cmfs.shape
        }

        self._mean = np.mean(self.reflectances, axis=0)
        self._XYZ_mu = msds_to_XYZ_integration(self._mean, **settings) / 100

        matrix_data = self.reflectances - self._mean
        matrix_covariance = np.dot(np.transpose(matrix_data), matrix_data)
        _eigenvalues, eigenvectors = np.linalg.eigh(matrix_covariance)
        self._basis_functions = np.transpose(eigenvectors[:, -3:])

        self._M = np.transpose(
            msds_to_XYZ_integration(self._basis_functions, **settings) / 100)

    def reconstruct(self, XYZ):
        """
        Reconstructs the reflectance for the given *CIE XYZ* tristimulus
        values.

        Parameters
        ----------
        XYZ : ndarray, (3,)
            *CIE XYZ* tristimulus values to recover the spectral distribution
            from.

        Returns
        -------
        SpectralDistribution
            Recovered spectral distribution.
        """

        weights = np.dot(np.linalg.inv(self._M), XYZ - self._XYZ_mu)
        reflectance = np.dot(weights, self._basis_functions) + self._mean
        reflectance = np.clip(reflectance, 0, 1)

        return SpectralDistribution(reflectance, self._cmfs.wavelengths)

    def reconstruction_error(self):
        """
        Returns the reconstruction error of the data. The error is computed by
        reconstructing the reflectances for the reference *CIE XYZ* tristimulus
        values using PCA and, comparing the reconstructed reflectances against
        the reference reflectances.

        Returns
        -------
        error : float
            The reconstruction error for the data.

        Notes
        -----
        -   The reconstruction error is cached upon being computed and thus is
            only computed once per node.
        """

        if self._reconstruction_error is not None:
            return self._reconstruction_error

        self.PCA()

        error = 0
        for i in range(len(self)):
            sd = self._reflectances[i, :]
            XYZ = self._XYZ[i, :]
            recovered_sd = self.reconstruct(XYZ)
            error += np.sum((sd - recovered_sd.values) ** 2)

        self._reconstruction_error = error

        return error


class Node_Otsu2018(Node):
    """
    Represents a node in a :meth:`colour.recovery.Tree_Otsu2018` class instance
    node tree.

    Parameters
    ----------
    parent : Node, optional
        Parent of the node.
    children : Node, optional
        Children of the node.
    data : Data_Otsu2018
        The colour data belonging to this node.

    Attributes
    ----------
    -   :attr:`~colour.recovery.otsu2018.Node.partition_axis`
    -   :attr:`~colour.recovery.otsu2018.Node.row`

    Methods
    -------
    -   :meth:`~colour.recovery.otsu2018.Node.__init__`
    -   :meth:`~colour.recovery.otsu2018.Node.split`
    -   :meth:`~colour.recovery.otsu2018.Node.minimise`
    -   :meth:`~colour.recovery.otsu2018.Node.leaf_reconstruction_error`
    -   :meth:`~colour.recovery.otsu2018.Node.branch_reconstruction_error`
    """

    def __init__(self, parent=None, children=None, data=None):
        super(Node_Otsu2018, self).__init__(
            parent=parent, children=children, data=data)

        self._partition_axis = None
        self._best_partition = None

    @property
    def partition_axis(self):
        """
        Getter property for the node partition axis.

        Returns
        -------
        PartitionAxis
            Node partition axis.
        """

        return self._partition_axis

    @property
    def row(self):
        """
        Getter property for the node row for the selector array.

        Returns
        -------
        list
            Node row for the selector array.
        """

        return [
            self._partition_axis.direction,
            self._partition_axis.origin,
        ] + self._children

    def split(self, children, axis):
        """
        Converts the leaf node into an inner node using given children and
        partition axis.

        Parameters
        ----------
        children : tuple
            Tuple of two :class:`colour.recovery.otsu2018.Node` class
            instances.
        axis : PartitionAxis
            Partition axis.
        """

        self.data = None
        self.children = children

        self._best_partition = None
        self._partition_axis = axis

    def minimise(self, minimum_cluster_size):
        """
        Finds the best partition for the node that minimises the leaf
        reconstruction error.

        Parameters
        ----------
        minimum_cluster_size : int
            Smallest acceptable cluster size. It must be at least 3 or the
            *Principal Component Analysis* (PCA) is not be possible.

        Returns
        -------
        partition : tuple
            Nodes created by splitting a node with a given partition.
        axis : PartitionAxis
            Horizontal or vertical line, partitioning the 2D space in
            two half-planes.
        partition_error : float
            Partition error
        """

        if self._best_partition is not None:
            return self._best_partition

        leaf_error = self.leaf_reconstruction_error()
        best_error = None

        with tqdm(total=2 * len(self.data)) as progress:
            for direction in [0, 1]:
                for i in range(len(self.data)):
                    progress.update()

                    axis = PartitionAxis(
                        self.data.origin(i, direction), direction)
                    data_lesser, data_greater = self.data.partition(axis)

                    if np.any(
                            np.array([
                                len(data_lesser),
                                len(data_greater),
                            ]) < minimum_cluster_size):
                        continue

                    lesser = Node_Otsu2018(data=data_lesser)
                    lesser.data.PCA()

                    greater = Node_Otsu2018(data=data_greater)
                    greater.data.PCA()

                    partition_error = (lesser.leaf_reconstruction_error() +
                                       greater.leaf_reconstruction_error())

                    partition = [lesser, greater]

                    if partition_error >= leaf_error:
                        continue

                    if best_error is None or partition_error < best_error:
                        self._best_partition = (partition, axis,
                                                partition_error)

        if self._best_partition is None:
            raise RuntimeError('Could not find the best partition!')

        return self._best_partition

    def leaf_reconstruction_error(self):
        """
        Returns the reconstruction error of the node data. The error is
        computed by reconstructing the reflectances for the data reference
        *CIE XYZ* tristimulus values using PCA and, comparing the reconstructed
        reflectances against the data reference reflectances.

        Returns
        -------
        error : float
            The reconstruction errors summation for the node data.
        """

        return self.data.reconstruction_error()

    def branch_reconstruction_error(self):
        """
        Computes the reconstruction error for all the leaves data connected to
        the node or its children, i.e. the reconstruction errors summation for
        all the leaves in the branch.

        Returns
        -------
        error : float
            Reconstruction errors summation for all the leaves data in the
            branch.
        """

        if self.is_leaf():
            return self.leaf_reconstruction_error()
        else:
            return np.sum([
                child.branch_reconstruction_error() for child in self.children
            ])


class Tree_Otsu2018(Node_Otsu2018):
    """
    A sub-class of :class:`colour.recovery.otsu2018.Node` class representing
    the root node of a tree containing information shared with all the nodes,
    such as the standard observer colour matching functions and the illuminant,
    if any is used.

    Global operations involving the entire tree, such as optimisation and
    conversion to dataset, are implemented in this sub-class.

    Parameters
    ----------
    reflectances : MultiSpectralDistributions
        Reference reflectances of the *n* reference colours to use for
        optimisation.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.

    Attributes
    ----------
    -   :attr:`~colour.recovery.Tree_Otsu2018.reflectances`
    -   :attr:`~colour.recovery.Tree_Otsu2018.cmfs`
    -   :attr:`~colour.recovery.Tree_Otsu2018.illuminant`

    Methods
    -------
    -   :meth:`~colour.recovery.otsu2018.Tree_Otsu2018.__init__`
    -   :meth:`~colour.recovery.otsu2018.Tree_Otsu2018.__str__`
    -   :meth:`~colour.recovery.otsu2018.Tree_Otsu2018.optimise`
    -   :meth:`~colour.recovery.otsu2018.Tree_Otsu2018.to_dataset`

    References
    ----------
    :cite:`Otsu2018`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> from colour import MSDS_CMFS, SDS_COLOURCHECKERS, SDS_ILLUMINANTS
    >>> from colour.colorimetry import sds_and_msds_to_msds
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> reflectances = sds_and_msds_to_msds(
    ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
    ... )
    >>> node_tree = Tree_Otsu2018(reflectances, cmfs, illuminant)
    >>> node_tree.optimise(iterations=2, print_callable=lambda x: x)
    >>> dataset = node_tree.to_dataset()
    >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
    ...                     'resources', 'ColorChecker_Otsu2018.npz')
    >>> dataset.write(path) # doctest: +SKIP
    >>> dataset = Dataset_Otsu2018() # doctest: +SKIP
    >>> dataset.read(path) # doctest: +SKIP
    >>> sd = XYZ_to_sd_Otsu2018(XYZ, cmfs, illuminant, dataset)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 360.        ,    0.0651341...],
                          [ 370.        ,    0.0651341...],
                          [ 380.        ,    0.0651341...],
                          [ 390.        ,    0.0749684...],
                          [ 400.        ,    0.0815578...],
                          [ 410.        ,    0.0776439...],
                          [ 420.        ,    0.0721897...],
                          [ 430.        ,    0.0649064...],
                          [ 440.        ,    0.0567185...],
                          [ 450.        ,    0.0484685...],
                          [ 460.        ,    0.0409768...],
                          [ 470.        ,    0.0358964...],
                          [ 480.        ,    0.0307857...],
                          [ 490.        ,    0.0270148...],
                          [ 500.        ,    0.0273773...],
                          [ 510.        ,    0.0303157...],
                          [ 520.        ,    0.0331285...],
                          [ 530.        ,    0.0363027...],
                          [ 540.        ,    0.0425987...],
                          [ 550.        ,    0.0513442...],
                          [ 560.        ,    0.0579256...],
                          [ 570.        ,    0.0653850...],
                          [ 580.        ,    0.0929522...],
                          [ 590.        ,    0.1600326...],
                          [ 600.        ,    0.2586159...],
                          [ 610.        ,    0.3701242...],
                          [ 620.        ,    0.4702243...],
                          [ 630.        ,    0.5396261...],
                          [ 640.        ,    0.5737561...],
                          [ 650.        ,    0.590848 ...],
                          [ 660.        ,    0.5935371...],
                          [ 670.        ,    0.5923295...],
                          [ 680.        ,    0.5956326...],
                          [ 690.        ,    0.5982513...],
                          [ 700.        ,    0.6017904...],
                          [ 710.        ,    0.6016419...],
                          [ 720.        ,    0.5996892...],
                          [ 730.        ,    0.6000018...],
                          [ 740.        ,    0.5964443...],
                          [ 750.        ,    0.5868181...],
                          [ 760.        ,    0.5860973...],
                          [ 770.        ,    0.5614878...],
                          [ 780.        ,    0.5289331...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    def __init__(self, reflectances, cmfs=None, illuminant=None):
        super(Tree_Otsu2018, self).__init__()

        self._cmfs, self._illuminant = handle_spectral_arguments(
            cmfs, illuminant, shape_default=SPECTRAL_SHAPE_OTSU2018)

        self._reflectances = np.transpose(
            reshape_msds(reflectances, self._cmfs.shape).values)

        self.data = Data_Otsu2018(self._reflectances, self._cmfs,
                                  self._illuminant)

    @property
    def reflectances(self):
        """
        Getter property for the reference reflectances.

        Returns
        -------
        ndarray
            Reference reflectances.
        """

        return self._reflectances

    @property
    def cmfs(self):
        """
        Getter property for the standard observer colour matching functions.

        Returns
        -------
        XYZ_ColourMatchingFunctions
            Standard observer colour matching functions.
        """

        return self._cmfs

    @property
    def illuminant(self):
        """
        Getter property for the illuminant.

        Returns
        -------
        SpectralDistribution
            Illuminant.
        """

        return self._illuminant

    def optimise(self,
                 iterations=8,
                 minimum_cluster_size=None,
                 print_callable=print):
        """
        Optimises the tree by repeatedly performing optimal partitioning of the
        nodes, creating a tree that minimises the total reconstruction error.

        Parameters
        ----------
        iterations : int, optional
            Maximum number of splits. If the dataset is too small, this number
            might not be reached. The default is to create 8 clusters, like in
            :cite:`Otsu2018`.
        minimum_cluster_size : int, optional
            Smallest acceptable cluster size. By default, it is chosen
            automatically, based on the size of the dataset and desired number
            of clusters. It must be at least 3 or the
            *Principal Component Analysis* (PCA) is not be possible.
        print_callable : callable, optional
            Callable used to print progress and diagnostic information.

        Examples
        --------
        >>> from colour.colorimetry import sds_and_msds_to_msds
        >>> from colour import MSDS_CMFS, SDS_COLOURCHECKERS, SDS_ILLUMINANTS
        >>> cmfs = (
        ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        ...     .copy().align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> reflectances = sds_and_msds_to_msds(
        ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
        ... )
        >>> node_tree = Tree_Otsu2018(reflectances, cmfs, illuminant)
        >>> node_tree.optimise(iterations=2)  # doctest: +ELLIPSIS
        ======================================================================\
=========
        *                                                                     \
        *
        *   "Otsu et al. (2018)" Tree Optimisation                            \
        *
        *                                                                     \
        *
        ======================================================================\
=========
        Initial branch error is: 4.8705353...
        <BLANKLINE>
        Iteration 1 of 2:
        <BLANKLINE>
        Optimising "Tree_Otsu2018#...(Data_Otsu2018(24 Reflectances))"...
        <BLANKLINE>
        Splitting "Tree_Otsu2018#...(Data_Otsu2018(24 Reflectances))" into \
"Node_Otsu2018#...(Data_Otsu2018(10 Reflectances))" and \
"Node_Otsu2018#...(Data_Otsu2018(14 Reflectances))" along \
"PartitionAxis(horizontal partition at y = 0.3240945...)".
        Error is reduced by 0.0054840... and is now 4.8650513..., 99.9% of \
the initial error.
        <BLANKLINE>
        Iteration 2 of 2:
        <BLANKLINE>
        Optimising "Node_Otsu2018#...(Data_Otsu2018(10 Reflectances))"...
        Optimisation failed: Could not find the best partition!
        Optimising "Node_Otsu2018#...(Data_Otsu2018(14 Reflectances))"...
        <BLANKLINE>
        Splitting "Node_Otsu2018#...(Data_Otsu2018(14 Reflectances))" into \
"Node_Otsu2018#...(Data_Otsu2018(7 Reflectances))" and \
"Node_Otsu2018#...(Data_Otsu2018(7 Reflectances))" along \
"PartitionAxis(horizontal partition at y = 0.3600663...)".
        Error is reduced by 0.9681059... and is now 3.8969453..., 80.0% of \
the initial error.
        Tree optimisation is complete!
        >>> print(node_tree.render())  # doctest: +ELLIPSIS
        |----"Tree_Otsu2018#..."
            |----"Node_Otsu2018#..."
            |----"Node_Otsu2018#..."
                |----"Node_Otsu2018#..."
                |----"Node_Otsu2018#..."
        <BLANKLINE>
        >>> len(node_tree)
        4
        """

        default_cluster_size = len(self.data) / iterations // 2
        minimum_cluster_size = max(
            (minimum_cluster_size
             if minimum_cluster_size is not None else default_cluster_size), 3)

        initial_branch_error = self.branch_reconstruction_error()

        message_box(
            '"Otsu et al. (2018)" Tree Optimisation',
            print_callable=print_callable)

        print_callable(
            'Initial branch error is: {0}'.format(initial_branch_error))

        best_leaf, best_partition, best_axis, partition_error = [None] * 4

        for i in range(iterations):
            print_callable('\nIteration {0} of {1}:\n'.format(
                i + 1, iterations))

            total_error = self.branch_reconstruction_error()
            optimised_total_error = None

            for leaf in self.leaves:
                print_callable('Optimising "{0}"...'.format(leaf))

                try:
                    partition, axis, partition_error = leaf.minimise(
                        minimum_cluster_size)
                except RuntimeError as error:
                    print_callable('Optimisation failed: {0}'.format(error))
                    continue

                new_total_error = (
                    total_error - leaf.leaf_reconstruction_error() +
                    partition_error)

                if (optimised_total_error is None or
                        new_total_error < optimised_total_error):
                    optimised_total_error = new_total_error
                    best_axis = axis
                    best_leaf = leaf
                    best_partition = partition

            if optimised_total_error is None:
                print_callable('\nNo further improvement is possible!\n'
                               'Terminating at iteration {0}.\n'.format(i))
                break

            print_callable(
                '\nSplitting "{0}" into "{1}" and "{2}" along "{3}".'.format(
                    best_leaf, best_partition[0], best_partition[1],
                    best_axis))

            print_callable(
                'Error is reduced by {0} and is now {1}, '
                '{2:.1f}% of the initial error.'.format(
                    leaf.leaf_reconstruction_error() - partition_error,
                    optimised_total_error,
                    100 * optimised_total_error / initial_branch_error))

            best_leaf.split(best_partition, best_axis)

        print_callable('Tree optimisation is complete!')

    def to_dataset(self):
        """
        Creates a :class:`colour.recovery.Dataset_Otsu2018` class instance
        based on data stored in the tree.

        The dataset can then be saved to disk or used to recover reflectance
        with :func:`colour.recovery.XYZ_to_sd_Otsu2018` definition.

        Returns
        -------
        Dataset_Otsu2018
            The dataset object.

        Examples
        --------
        >>> from colour.colorimetry import sds_and_msds_to_msds
        >>> from colour.characterisation import SDS_COLOURCHECKERS
        >>> reflectances = sds_and_msds_to_msds(
        ...     SDS_COLOURCHECKERS['ColorChecker N Ohta'].values()
        ... )
        >>> node_tree = Tree_Otsu2018(reflectances)
        >>> node_tree.optimise(iterations=2, print_callable=lambda x: x)
        >>> node_tree.to_dataset()  # doctest: +ELLIPSIS
        <colour.recovery.otsu2018.Dataset_Otsu2018 object at 0x...>
        """

        basis_functions = as_float_array(
            [leaf._data.basis_functions for leaf in self.leaves])

        means = as_float_array([leaf._data.mean for leaf in self.leaves])

        if len(self.children) == 0:
            selector_array = zeros(4)
        else:

            def add_rows(node, data=None):
                """
                Add rows for given node and its children.
                """

                if data is None:
                    data = {'rows': [], 'node_to_leaf_id': {}, 'leaf_id': 0}

                if node.is_leaf():
                    data['node_to_leaf_id'][node] = data['leaf_id']
                    data['leaf_id'] += 1
                    return

                data['node_to_leaf_id'][node] = -len(data['rows'])
                data['rows'].append(node.row)

                for child in node.children:
                    add_rows(child, data)

                return data

            data = add_rows(self)
            rows = data['rows']

            for i, row in enumerate(rows):
                for j in (2, 3):
                    rows[i][j] = data['node_to_leaf_id'][row[j]]

            selector_array = as_float_array(rows)

        return Dataset_Otsu2018(
            self._cmfs.shape,
            basis_functions,
            means,
            selector_array,
        )
