Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 312 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/spectrum.py

    -   Line 1131 : # TODO: Provide support for fractional interval like 0.1, etc...


-   colour/colorimetry/tristimulus.py

    -   Line 769 : # TODO: Investigate code vectorisation.


-   colour/colorimetry/tests/test_spectrum.py

    -   Line 1486 : # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version. Skipping tests because of "Scipy" 0.19.0 interpolation code changes.
    -   Line 1695 : # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version. Skipping tests because of "Scipy" 0.19.0 interpolation code changes.


-   colour/appearance/ciecam02.py

    -   Line 298 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 292 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 420 : # TODO: Implement hue quadrature & composition computation.
    -   Line 451 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 265 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 267 : # TODO: Implement hue quadrature & composition computation.
    -   Line 279 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 325 : # TODO: Implement hue composition computation.


-   colour/appearance/tests/test_cam16.py

    -   Line 37 : # TODO: The current fixture data is generated from direct computations using our model implementation. We have asked ground truth data to Li et al. (2016) and will update the "cam16.csv" file accordingly whenever we receive it.


-   colour/io/tests/test_image.py

    -   Line 210 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/io/luts/lut.py

    -   Line 129 : # TODO: Re-enable when dropping Python 2.7. pylint: disable=E1121
    -   Line 2012 : # TODO: Implement support for non-uniform domain, e.g. "cinespace" LUTs.


-   colour/graph/conversion.py

    -   Line 927 : # TODO: Remove the following warning whenever the automatic colour conversion graph implementation is considered stable.


-   colour/models/rgb/derivation.py

    -   Line 217 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/rgb_colourspace.py

    -   Line 541 : # TODO: Revisit for potential behaviour / type checking.
    -   Line 574 : # TODO: Revisit for potential behaviour / type checking.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 279 : # TODO: Simplify that monster.


-   colour/models/rgb/transfer_functions/tests/test__init__.py

    -   Line 39 : # TODO: Use "assertWarns" when dropping Python 2.7.
    -   Line 56 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/utilities/verbose.py

    -   Line 572 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library.


-   colour/utilities/common.py

    -   Line 601 : # TODO: Remove when dropping Python 2.7.


-   colour/utilities/array.py

    -   Line 186 : # TODO: Change to "DEFAULT_INT_DTYPE" when and if https://github.com/numpy/numpy/issues/11956 is addressed.


-   colour/utilities/tests/test_deprecation.py

    -   Line 317 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/plotting/models.py

    -   Line 1569 : # TODO: Filter appropriate colour models.


-   colour/plotting/common.py

    -   Line 666 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 783 : # TODO: Consider using "MutableMapping" here.


-   colour/characterisation/correction.py

    -   Line 225 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1081 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 384 : # TODO: Check for interpolator capabilities.
    -   Line 449 : # TODO: Check for extrapolator capabilities.


-   colour/continuous/multi_signals.py

    -   Line 1322 : # TODO: Implement support for Signal class passing.


-   colour/continuous/tests/test_multi_signal.py

    -   Line 112 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/continuous/tests/test_signal.py

    -   Line 101 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/volume/rgb.py

    -   Line 306 : # TODO: Investigate for generator yielding directly a ndarray.


-   colour/algebra/tests/test_interpolation.py

    -   Line 532 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.


-   colour/algebra/tests/test_random.py

    -   Line 61 : # TODO: Use "assertWarns" when dropping Python 2.7.

About
-----

| **Colour** by Colour Developers
| Copyright © 2013-2020 – Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
