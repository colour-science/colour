Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 337 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/spectrum.py

    -   Line 1093 : # TODO: Provide support for fractional interval like 0.1, etc...


-   colour/colorimetry/tristimulus.py

    -   Line 757 : # TODO: Investigate code vectorisation.


-   colour/colorimetry/blackbody.py

    -   Line 580 : # TODO: Remove warning when deemed appropriate.


-   colour/colorimetry/tests/test_spectrum.py

    -   Line 1459 : # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version. Skipping tests because of "Scipy" 0.19.0 interpolation code changes.
    -   Line 1665 : # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version. Skipping tests because of "Scipy" 0.19.0 interpolation code changes.


-   colour/appearance/ciecam02.py

    -   Line 313 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 292 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 414 : # TODO: Implement hue quadrature & composition computation.
    -   Line 445 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 264 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 266 : # TODO: Implement hue quadrature & composition computation.
    -   Line 278 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 320 : # TODO: Implement hue composition computation.


-   colour/appearance/tests/test_cam16.py

    -   Line 37 : # TODO: The current fixture data is generated from direct computations using our model implementation. We have asked ground truth data to Li et al. (2016) and will update the "cam16.csv" file accordingly whenever we receive it.


-   colour/recovery/otsu2018.py

    -   Line 849 : # TODO: Python 3 "yield from child.leaves".


-   colour/io/image.py

    -   Line 64 : # TODO: Overhaul by using "np.sctypeDict".


-   colour/io/tests/test_image.py

    -   Line 210 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/io/luts/lut.py

    -   Line 131 : # TODO: Re-enable when dropping Python 2.7. pylint: disable=E1121


-   colour/graph/conversion.py

    -   Line 953 : # TODO: Remove the following warning whenever the automatic colour conversion graph implementation is considered stable.


-   colour/models/rgb/derivation.py

    -   Line 211 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/rgb_colourspace.py

    -   Line 559 : # TODO: Revisit for potential behaviour / type checking.
    -   Line 592 : # TODO: Revisit for potential behaviour / type checking.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 279 : # TODO: Simplify that monster.


-   colour/models/rgb/transfer_functions/tests/test__init__.py

    -   Line 39 : # TODO: Use "assertWarns" when dropping Python 2.7.
    -   Line 56 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/utilities/verbose.py

    -   Line 609 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library.


-   colour/utilities/common.py

    -   Line 709 : # TODO: Remove when dropping Python 2.7.


-   colour/utilities/array.py

    -   Line 81 : # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is addressed.
    -   Line 234 : # TODO: Change to "DEFAULT_INT_DTYPE" when and if https://github.com/numpy/numpy/issues/11956 is addressed.
    -   Line 370 : # TODO: Investigate behaviour on Windows.


-   colour/utilities/tests/test_deprecation.py

    -   Line 317 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/plotting/models.py

    -   Line 1725 : # TODO: Filter appropriate colour models.


-   colour/plotting/graph.py

    -   Line 72 : # TODO: Investigate API to trigger the conversion graph build.


-   colour/plotting/common.py

    -   Line 666 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 791 : # TODO: Consider using "MutableMapping" here.


-   colour/characterisation/aces_it.py

    -   Line 322 : # TODO: Remove when removing the "colour.sd_blackbody" definition warning.


-   colour/characterisation/correction.py

    -   Line 353 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1078 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 389 : # TODO: Check for interpolator capabilities.
    -   Line 454 : # TODO: Check for extrapolator capabilities.


-   colour/continuous/multi_signals.py

    -   Line 1354 : # TODO: Implement support for Signal class passing.


-   colour/continuous/tests/test_multi_signal.py

    -   Line 112 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/continuous/tests/test_signal.py

    -   Line 102 : # TODO: Use "assertWarns" when dropping Python 2.7.


-   colour/volume/rgb.py

    -   Line 300 : # TODO: Investigate for generator yielding directly a ndarray.


-   colour/algebra/tests/test_interpolation.py

    -   Line 534 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.


-   colour/algebra/tests/test_random.py

    -   Line 68 : # TODO: Use "assertWarns" when dropping Python 2.7.

About
-----

| **Colour** by Colour Developers
| Copyright © 2013-2020 – Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
