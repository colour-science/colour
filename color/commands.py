#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**commands.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Exposes various **Color** package objects for command line execution.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External Imports.
#**********************************************************************************************************************
import argparse
import ast
import functools
import pprint
import sys
import traceback
import numpy

#**********************************************************************************************************************
#***	Internal Imports.
#**********************************************************************************************************************
import color.chromaticAdaptation
import color.illuminants
import color.temperature
import color.transformations
import color.verbose
from color.globals.constants import Constants
# from color.temperature import Temperature

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "DEFAULT_ARGUMENTS_VALUES",
		   "ManualAction",
		   "ParseListAction",
		   "systemExit",
		   "commands",
		   "getCommandLineArguments",
		   "main"]

LOGGER = color.verbose.installLogger()

DEFAULT_ARGUMENTS_VALUES = {"x": 0.5,
							"y": 0.5,
							"Temperature": 6500,
							"Tint": 0,
							"XYZ Matrix": (0.5, 0.5, 0.5),
							"Source XYZ Matrix": (0.5, 0.5, 0.5),
							"Target XYZ Matrix": (0.5, 0.5, 0.5), }

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class ManualAction(argparse.Action):
	"""
	Handles conversion of '-m/--manual' argument in order to provide detailed manual with usage examples.
	"""

	def __call__(self, parser, namespace, values, option_string=None):
		"""
		Reimplements the :meth:`argparse.Action.__call__` method.

		:param parser: Parser.
		:type parser: object
		:param namespace: Namespace.
		:type namespace: object
		:param values: Values.
		:type values: object
		:param option_string: Option string.
		:type option_string: object
		"""

		print("""'color' Commands Manual

NAME
    color -- Implements 'Color' package command line support.

SYNOPSIS
    color [-v] [-h] [-m] [-d]
          [-x X]
          [-y Y]
          [--temperature TEMPERATURE]
          [--tint TINT]
          [--xyzMatrix XYZMATRIX]
          [--sourceXyzMatrix SOURCEXYZMATRIX]
          [--targetXyzMatrix TARGETXYZMATRIX]
          [--getTemperature]
          [--getChromaticityCoordinates]
          [--XYZ_to_xy]
          [--xy_to_XYZ]
          [--getChromaticAdaptationMatrix]
          [--displayDefaultArgumentsValues]
          [--displayWyszeckiRoberstonTable]

DESCRIPTION
    This tool implements relevant definitions and methods from 'Color' package for command line usage.

ARGUMENTS
    -v, --version
      'Displays application release version.'

    -h, --help
      'Displays this help message and exit. Please use -m/--manual for examples.'

    -m, --manual
      'Displays detailed manual with usage examples.'

    -d, --details
      'Displays detailed messages.'

    -x X
      'X Chromaticity Coordinate.'

    -y Y
      'Y Chromaticity Coordinate.'

    --temperature TEMPERATURE
      'Temperature.'

    --tint TINT
      'Tint.'

    --xyzMatrix XYZMATRIX
      ''CIE XYZ' matrix.'

    --sourceXyzMatrix SOURCEXYZMATRIX
      'Source 'CIE XYZ' matrix for 'Chromatic Adaptation' matrix calculations.'

    --targetXyzMatrix TARGETXYZMATRIX
      'Target 'CIE XYZ' matrix for 'Chromatic Adaptation' matrix calculations.'

    --calibrationIlluminant1 CALIBRATIONILLUMINANT1
      'Calibration Illuminant 1.'

    --calibrationIlluminant2 CALIBRATIONILLUMINANT2
      'Calibration Illuminant 2.'

    --getTemperature
      'Returns correlated color temperature and tint from given chromaticity coordinates.'

    --getChromaticityCoordinates
      'Returns chromaticity coordinates from given correlated color temperature and tint values.'

    --XYZ_to_xy
      'Returns chromaticity coordinates from given 'CIE XYZ' matrix values.'

    --xy_to_XYZ
      'Returns 'CIE XYZ' matrix values from given chromaticity coordinates.'

    --getChromaticAdaptationMatrix
      'Returns Chromatic Adaptation matrix using given 'CIE XYZ' source and target matrices values.'

    --displayDefaultArgumentsValues
      'Displays default arguments values.'

    --displayWyszeckiRoberstonTable
      'Displays Wyszecki & Roberston table.'

EXAMPLES
    All commands have various display capabilities triggered by the '-d/--details' argument:

        > color --getTemperature -x 0.25 -y 0.25 -d
        ********************************************************************************
        Color - 0.0.1 | Starting commands processing ...
        ********************************************************************************
        Correlated color temperature and tint values from chromaticity coordinates:
        (x, y): (0.25, 0.25)
        (Temperature, Tint): (28847.66573353418, 2.0587866255277527)
        ********************************************************************************
        Color - 0.0.1 | Ending commands processing ...
        ********************************************************************************

    Retrieving correlated color temperature and tint values from given chromaticity coordinates:

        > color --getTemperature -x 0.25 -y 0.25
        (28847.66573353418, 2.0587866255277527)

    Retrieving chromaticity coordinates from given correlated color temperature and tint values:

        > color --getChromaticityCoordinates --temperature 6000 --tint 5
        (0.32186800157610984, 0.33459329446365205)

    Retrieving chromaticity coordinates from given 'CIE XYZ' matrix values:

        > color --XYZ_to_xy --xyzMatrix "(0.5, 0.5, 0.5)"
        (0.33333333333333331, 0.33333333333333331)

    Retrieving the 'CIE XYZ' matrix values from given chromaticity coordinates:

        > color --xy_to_XYZ -x 0.25 -y 0.25
        matrix([[ 1.],
                [ 1.],
                [ 2.]])

    Retrieving the 'Chromatic Adaptation' matrix values from given 'CIE XYZ' matrices values:

        > color --color.chromaticAdaptation.getChromaticAdaptationMatrix --sourceXyzMatrix "(0.5, 0.5, 0.5)" --targetXyzMatrix "(0.25, 0.25, 0.25)"
        matrix([[  5.00000000e-01,  -7.19910243e-17,   0.00000000e+00],
                [  2.02881332e-17,   5.00000000e-01,   0.00000000e+00],
                [ -6.50521303e-19,   0.00000000e+00,   5.00000000e-01]])
""")

		sys.exit(0)

class ParseListAction(argparse.Action):
	"""
	Handles conversion of various arguments in order to convert strings to lists.
	"""

	def __call__(self, parser, namespace, values, option_string=None):
		"""
		Reimplements the :meth:`argparse.Action.__call__` method.

		:param parser: Parser.
		:type parser: object
		:param namespace: Namespace.
		:type namespace: object
		:param values: Values.
		:type values: object
		:param option_string: Option string.
		:type option_string: object
		"""

		setattr(namespace, self.dest, ast.literal_eval(values))

def systemExit(object):
	"""
	Handles proper system exit in case of critical exception.

	:param object: Object to decorate.
	:type object: object
	:return: Object.
	:rtype: object
	"""

	@functools.wraps(object)
	def systemExitWrapper(*args, **kwargs):
		"""
		Handles proper system exit in case of critical exception.

		:param \*args: Arguments.
		:type \*args: \*
		:param \*\*kwargs: Keywords arguments.
		:type \*\*kwargs: \*\*
		"""

		try:
			if object(*args, **kwargs):
				sys.exit()
		except Exception as error:
			traceback.print_exc()
			sys.exit(1)

	return systemExitWrapper

def commands(args):
	"""
	Implements relevant definitions and methods from **Color** package for independent usage.

	:param args: Arguments namespace.
	:type args: Namespace
	:return: Definition success.
	:rtype: bool
	"""

	title = "{0} - {1}".format(Constants.applicationName, Constants.version)

	verbose = args.details

	if verbose:
		LOGGER.info(Constants.loggingSeparators)
		LOGGER.info("{0} | Starting commands processing ...".format(Constants.applicationName))

	if args.version:
		if not verbose:
			print(title)
		else:
			LOGGER.info("Application release version: {0}".format(title))

	if args.getTemperature:
		cct, tint = Temperature().setChromaticityCoordinates((args.x, args.y))
		if not verbose:
			print((cct, tint))
		else:
			LOGGER.info("Correlated color temperature and tint values from chromaticity coordinates:")
			LOGGER.info("(x, y):\n{0}".format((args.x, args.y)))
			LOGGER.info("(Temperature, Tint):\n{0}".format((cct, tint)))

	if args.getChromaticityCoordinates:
		chromaticityCoordinates = Temperature(args.temperature,
											  args.tint).getChromaticityCoordinates()
		if not verbose:
			print(chromaticityCoordinates)
		else:
			LOGGER.info("Chromaticity coordinates from correlated color temperature and tint values:")
			LOGGER.info("(Temperature, Tint):\n{0}".format((args.temperature, args.tint)))
			LOGGER.info("(x, y):\n{0}".format(chromaticityCoordinates))

	if args.XYZ_to_xy:
		chromaticityCoordinates = color.transformations.XYZ_to_xy(args.xyzMatrix)
		if not verbose:
			print(chromaticityCoordinates)
		else:
			LOGGER.info("Chromaticity coordinates from 'CIE XYZ' matrix:")
			LOGGER.info("'CIE XYZ' matrix:\n{0}".format(args.xyzMatrix))
			LOGGER.info("(x, y):\n{0}".format(chromaticityCoordinates))

	if args.xy_to_XYZ:
		xyzMatrix = color.transformations.xy_to_XYZ((args.x, args.y))
		if not verbose:
			print(repr(xyzMatrix))
		else:
			LOGGER.info("'CIE XYZ' matrix from chromaticity coordinates:")
			LOGGER.info("(x, y):\n{0}".format((args.x, args.y)))
			LOGGER.info("'CIE XYZ' matrix:\n{0}".format(repr(xyzMatrix)))

	if args.getChromaticAdaptationMatrix:
		chromaticAdaptationMatrix = color.chromaticAdaptation.getChromaticAdaptationMatrix(
			numpy.matrix(args.sourceXyzMatrix).reshape((3, 1)),
			numpy.matrix(args.targetXyzMatrix).reshape((3, 1)))
		if not verbose:
			print(repr(chromaticAdaptationMatrix))
		else:
			LOGGER.info("'Chromatic adaptation' matrix from given 'CIE XYZ' matrices values:")
			LOGGER.info("Source 'CIE XYZ' matrix:\n{0}".format(args.sourceXyzMatrix))
			LOGGER.info("Target 'CIE XYZ' matrix:\n{0}".format(args.targetXyzMatrix))
			LOGGER.info("'Chromatic adaptation' matrix:\n{0}".format(repr(chromaticAdaptationMatrix)))

	if args.displayDefaultArgumentsValues:
		if not verbose:
			pprint.pprint(DEFAULT_ARGUMENTS_VALUES)
		else:
			LOGGER.info("Default arguments values:")
			pprint.pformat(DEFAULT_ARGUMENTS_VALUES)

	if args.displayWyszeckiRoberstonTable:
		if not verbose:
			pprint.pprint(color.temperature.WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA)
		else:
			LOGGER.info("(Reciprocal Megakelvin, CIE 1960 Chromaticity Coordinates 'u', CIE 1960 Chromaticity Coordinates 'v', Slope)")
			pprint.pformat(color.temperature.WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA)

	if verbose:
		LOGGER.info("{0} | Ending commands processing ...".format(Constants.applicationName))
		LOGGER.info(Constants.loggingSeparators)

	return True

def getCommandLineArguments():
	"""
	Retrieves command line arguments.

	:return: Namespace.
	:rtype: Namespace
	"""

	parser = argparse.ArgumentParser(add_help=False)

	parser.add_argument("-v",
						"--version",
						action="store_true",
						dest="version",
						help="'Displays application release version.'")

	parser.add_argument("-h",
						"--help",
						action="help",
						help="'Displays this help message and exit. Please use -m/--manual for examples.'")

	parser.add_argument("-m",
						"--manual",
						action=ManualAction,
						help="'Displays detailed manual with usage examples.'",
						nargs=0)

	parser.add_argument("-d",
						"--details",
						action="store_true",
						dest="details",
						help="'Displays detailed messages.'")

	parser.add_argument("-x",
						action="store",
						type=float,
						dest="x",
						default=DEFAULT_ARGUMENTS_VALUES.get("x"),
						help="'X Chromaticity Coordinate.'")

	parser.add_argument("-y",
						action="store",
						type=float,
						dest="y",
						default=DEFAULT_ARGUMENTS_VALUES.get("y"),
						help="'Y Chromaticity Coordinate.'")

	parser.add_argument("--temperature",
						action="store",
						type=float,
						dest="temperature",
						default=DEFAULT_ARGUMENTS_VALUES.get("Temperature"),
						help="'Temperature.'")

	parser.add_argument("--tint",
						action="store",
						type=float,
						dest="tint",
						default=DEFAULT_ARGUMENTS_VALUES.get("Tint"),
						help="'Tint.'")

	parser.add_argument("--xyzMatrix",
						action=ParseListAction,
						dest="xyzMatrix",
						default=DEFAULT_ARGUMENTS_VALUES.get("XYZ Matrix"),
						help="'XYZ matrix.'")

	parser.add_argument("--sourceXyzMatrix",
						action=ParseListAction,
						dest="sourceXyzMatrix",
						default=DEFAULT_ARGUMENTS_VALUES.get("Source XYZ Matrix"),
						help="'Source 'CIE XYZ' matrix for 'Chromatic Adaptation' matrix calculations.'")

	parser.add_argument("--targetXyzMatrix",
						action=ParseListAction,
						dest="targetXyzMatrix",
						default=DEFAULT_ARGUMENTS_VALUES.get("Target XYZ Matrix"),
						help="'Target 'CIE XYZ' matrix for 'Chromatic Adaptation' matrix calculations.'")

	parser.add_argument("--getTemperature",
						action="store_true",
						dest="getTemperature",
						help="'Returns correlated color temperature and tint from given chromaticity coordinates.'")

	parser.add_argument("--getChromaticityCoordinates",
						action="store_true",
						dest="getChromaticityCoordinates",
						help="'Returns chromaticity coordinates from given correlated color temperature and tint values.'")

	parser.add_argument("--XYZ_to_xy",
						action="store_true",
						dest="XYZ_to_xy",
						help="'Returns chromaticity coordinates from given 'CIE XYZ' matrix values.'")

	parser.add_argument("--xy_to_XYZ",
						action="store_true",
						dest="xy_to_XYZ",
						help="'Returns 'CIE XYZ' matrix values from given chromaticity coordinates.'")

	parser.add_argument("--getChromaticAdaptationMatrix",
						action="store_true",
						dest="getChromaticAdaptationMatrix",
						help="'Returns Chromatic Adaptation matrix using given 'CIE XYZ' source and target matrices values.'")

	parser.add_argument("--displayDefaultArgumentsValues",
						action="store_true",
						dest="displayDefaultArgumentsValues",
						help="'Displays default arguments values.'")

	parser.add_argument("--displayWyszeckiRoberstonTable",
						action="store_true",
						dest="displayWyszeckiRoberstonTable",
						help="'Displays Wyszecki & Roberston table.'")

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	return parser.parse_args()

@systemExit
def main():
	"""
	Starts the application.

	:return: Definition success.
	:rtype: bool
	"""

	return commands(getCommandLineArguments())

if __name__ == "__main__":
	main()
