#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.phenomenon.rayleigh` module.
"""

from __future__ import division, unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.phenomenon.rayleigh import (
    air_refraction_index_penndorf1957,
    air_refraction_index_edlen1966,
    air_refraction_index_peck1972,
    air_refraction_index_bodhaine1999,
    N2_depolarisation,
    O2_depolarisation,
    F_air_penndorf1957,
    F_air_young1981,
    F_air_bates1984,
    F_air_bodhaine1999,
    molecular_density,
    mean_molecular_weights,
    gravity_list1968)

from colour.phenomenon import scattering_cross_section, rayleigh_optical_depth

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestGetColourRenderingIndex']


class TestAirRefractionIndexPenndorf1957(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.air_refraction_index_penndorf1957`
    definition unit tests methods.
    """

    def test_air_refraction_index_penndorf1957(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.air_refraction_index_penndorf1957`
        definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_penndorf1957(0.360),
            1.0002853167951464,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_penndorf1957(0.555),
            1.000277729533864,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_penndorf1957(0.830),
            1.000274856640486,
            places=10)


class TestAirRefractionIndexEdlen1966(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.air_refraction_index_edlen1966`
    definition unit tests methods.
    """

    def test_air_refraction_index_edlen1966(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.air_refraction_index_edlen1966`
        definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_edlen1966(0.360),
            1.000285308809879,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_edlen1966(0.555),
            1.000277727690364,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_edlen1966(0.830),
            1.0002748622188347,
            places=10)


class TestAirRefractionIndexPeck1972(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.air_refraction_index_peck1972`
    definition unit tests methods.
    """

    def test_air_refraction_index_peck1972(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.air_refraction_index_peck1972`
        definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_peck1972(0.360),
            1.0002853102850557,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_peck1972(0.555),
            1.0002777265414837,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_peck1972(0.830),
            1.0002748591448039,
            places=10)


class TestAirRefractionIndexBodhaine1999(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.air_refraction_index_bodhaine1999`
    definition unit tests methods.
    """

    def test_air_refraction_index_bodhaine1999(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.air_refraction_index_bodhaine1999`
        definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.360),
            1.0002853102850557,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.555),
            1.0002777265414837,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.830),
            1.0002748591448039,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.360, 0),
            1.0002852640647895,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.555, 360),
            1.0002777355398236,
            places=10)

        self.assertAlmostEqual(
            air_refraction_index_bodhaine1999(0.830, 620),
            1.0002749066404641,
            places=10)


class TestN2Depolarisation(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.N2_depolarisation`
    definition unit tests methods.
    """

    def test_N2_depolarisation(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.N2_depolarisation`
        definition.
        """

        self.assertAlmostEqual(
            N2_depolarisation(0.360),
            1.036445987654321,
            places=7)

        self.assertAlmostEqual(
            N2_depolarisation(0.555),
            1.0350291372453535,
            places=7)

        self.assertAlmostEqual(
            N2_depolarisation(0.830),
            1.034460153868486,
            places=7)


class TestO2Depolarisation(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.O2_depolarisation`
    definition unit tests methods.
    """

    def test_O2_depolarisation(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.O2_depolarisation`
        definition.
        """

        self.assertAlmostEqual(
            O2_depolarisation(0.360),
            1.115307746532541,
            places=7)

        self.assertAlmostEqual(
            O2_depolarisation(0.555),
            1.1020225362010714,
            places=7)

        self.assertAlmostEqual(
            O2_depolarisation(0.830),
            1.0983155612690134,
            places=7)


class TestF_airPenndorf1957(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.F_air_penndorf1957`
    definition unit tests methods.
    """

    def test_F_air_penndorf1957(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.F_air_penndorf1957`
        definition.
        """

        self.assertEqual(F_air_penndorf1957(), 1.0608)


class TestF_airYoung1981(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.F_air_young1981`
    definition unit tests methods.
    """

    def test_F_air_young1981(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.F_air_young1981`
        definition.
        """

        self.assertEqual(F_air_young1981(), 1.0480)


class TestF_airBates1984(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.F_air_bates1984`
    definition unit tests methods.
    """

    def test_F_air_bates1984(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.F_air_bates1984`
        definition.
        """

        self.assertAlmostEqual(
            F_air_bates1984(0.360),
            1.051997277711708,
            places=7)

        self.assertAlmostEqual(
            F_air_bates1984(0.555),
            1.048153579718658,
            places=7)

        self.assertAlmostEqual(
            F_air_bates1984(0.830),
            1.0469470686005893,
            places=7)


class TestF_airBodhaine1999(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.F_air_bodhaine1999`
    definition unit tests methods.
    """

    def test_F_air_bodhaine1999(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.F_air_bodhaine1999`
        definition.
        """

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.360),
            1.125664021159081,
            places=7)

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.555),
            1.1246916702401561,
            places=7)

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.830),
            1.1243864557835395,
            places=7)

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.360, 0),
            1.0526297923139392,
            places=7)

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.555, 360),
            1.1279930150966895,
            places=7)

        self.assertAlmostEqual(
            F_air_bodhaine1999(0.830, 620),
            1.13577082243141,
            places=7)


class TestMolecularDensity(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.molecular_density`
    definition unit tests methods.
    """

    def test_molecular_density(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.molecular_density`
        definition.
        """

        self.assertAlmostEqual(
            molecular_density(200),
            3.669449208173649e+19,
            places=24)

        self.assertAlmostEqual(
            molecular_density(300),
            2.4462994721157665e+19,
            places=24)

        self.assertAlmostEqual(
            molecular_density(400),
            1.8347246040868246e+19,
            places=24)


class TestMeanMolecularWeights(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.mean_molecular_weights`
    definition unit tests methods.
    """

    def test_mean_molecular_weights(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.mean_molecular_weights`
        definition.
        """

        self.assertAlmostEqual(
            mean_molecular_weights(0),
            28.9595,
            places=7)

        self.assertAlmostEqual(
            mean_molecular_weights(360),
            28.964920015999997,
            places=7)

        self.assertAlmostEqual(
            mean_molecular_weights(620),
            28.968834471999998,
            places=7)


class TestGravityList1968(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.gravity_list1968`
    definition unit tests methods.
    """

    def test_gravity_list1968(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.gravity_list1968`
        definition.
        """

        self.assertAlmostEqual(
            gravity_list1968(0, 0),
            978.0356070576,
            places=7)

        self.assertAlmostEqual(
            gravity_list1968(45, 1500),
            980.1533438638013,
            places=7)

        self.assertAlmostEqual(
            gravity_list1968(48.8567, 35),
            980.9524178426182,
            places=7)


class TestScatteringCrossSection(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.scattering_cross_section`
    definition unit tests methods.
    """

    def test_scattering_cross_section(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.scattering_cross_section`
        definition.
        """

        self.assertAlmostEqual(
            scattering_cross_section(360 * 10e-8),
            2.7812892348020306e-26,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8),
            4.661330902337604e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(830 * 10e-8),
            9.12510035221888e-28,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 0),
            4.3465433368391025e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 360),
            4.675013461928133e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 620),
            4.707951639592975e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=200),
            2.245601437154005e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=300),
            5.05260323359651e-27,
            places=32)

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=400),
            8.98240574861602e-27,
            places=32)


class TestRayleighOpticalDepth(unittest.TestCase):
    """
    Defines
    :func:`colour.phenomenon.rayleigh.rayleigh_optical_depth`
    definition unit tests methods.
    """

    def test_rayleigh_optical_depth(self):
        """
        Tests
        :func:`colour.phenomenon.rayleigh.rayleigh_optical_depth`
        definition.
        """

        self.assertAlmostEqual(
            rayleigh_optical_depth(360 * 10e-8),
            0.5991013368480278,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8),
            0.10040701772896546,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(830 * 10e-8),
            0.019655847912113555,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 0),
            0.09364096434804903,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 360),
            0.10069860517689733,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 620),
            0.10139438226086347,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=200),
            0.0483711944156206,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=300),
            0.10883518743514632,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=400),
            0.1934847776624824,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=101325),
            0.10040701772896546,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=100325),
            0.0994160775095826,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=99325),
            0.09842513729019979,
            places=7)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=0, altitude=0),
            0.10040701772896546,
            places=10)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=45, altitude=1500),
            0.10019007653463426,
            places=10)

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=48.8567, altitude=35),
            0.10010846270542267,
            places=10)


if __name__ == '__main__':
    unittest.main()
