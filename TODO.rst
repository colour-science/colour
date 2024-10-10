Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 913 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/spectrum.py

    -   Line 1174 : # TODO: Provide support for fractional interval like 0.1, etc...


-   colour/colorimetry/tristimulus_values.py

    -   Line 1050 : # TODO: Investigate code vectorisation.


-   colour/appearance/ciecam02.py

    -   Line 369 : # TODO: Compute hue composition.


-   colour/appearance/ciecam16.py

    -   Line 322 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 308 : # TODO: Compute hue composition.


-   colour/appearance/hellwig2022.py

    -   Line 354 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 478 : # TODO: Implement hue quadrature & composition computation.
    -   Line 509 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 283 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 307 : # TODO: Implement hue quadrature & composition computation.
    -   Line 318 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 362 : # TODO: Implement hue composition computation.


-   colour/recovery/tests/test_jiang2013.py

    -   Line 66 : # TODO: Last eigen value seems to be very sensitive and produce differences on ARM.


-   colour/io/fichet2021.py

    -   Line 644 : # TODO: Implement support for integration of bi-spectral component.
    -   Line 651 : # TODO: Implement support for re-binning component with non-uniform interval.


-   colour/io/ctl.py

    -   Line 65 : # TODO: Reinstate coverage when "ctlrender" is trivially available cross-platform.


-   colour/io/tests/test_ocio.py

    -   Line 37 : # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/tests/test_ctl.py

    -   Line 39 : # TODO: Reinstate coverage when "ctlrender" is tivially available cross-platform.


-   colour/io/tests/test_image.py

    -   Line 322 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/models/rgb/derivation.py

    -   Line 230 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/tests/test_rgb_colourspace.py

    -   Line 340 : # TODO: Remove tests when dropping deprecated signature support.
    -   Line 541 : # TODO: Remove tests when dropping deprecated signature support.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 327 : # TODO: Simplify that monster.


-   colour/utilities/verbose.py

    -   Line 795 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library. NOTE: A few clauses are not reached and a few packages are not available during continuous integration and are thus ignored for coverage.


-   colour/utilities/network.py

    -   Line 587 : # TODO: Consider using an ordered set instead of a dict.
    -   Line 1064 : # TODO: Consider using ordered set.
    -   Line 1070 : # TODO: Consider using ordered set.
    -   Line 1907 : # TODO: Implement solid control flow based processing using a stack.


-   colour/utilities/array.py

    -   Line 573 : # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is addressed.
    -   Line 853 : # TODO: Investigate behaviour on Windows.
    -   Line 914 : # TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]" when Python 3.7 is dropped.


-   colour/plotting/models.py

    -   Line 1925 : # TODO: Filter appropriate colour models. NOTE: "dtype=object" is required for ragged array support in "Numpy" 1.24.0.


-   colour/plotting/graph.py

    -   Line 86 : # TODO: Investigate API to trigger the conversion graph build.


-   colour/plotting/common.py

    -   Line 871 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 987 : # TODO: Consider using "MutableMapping" here.


-   colour/characterisation/aces_it.py

    -   Line 395 : # TODO: Remove when removing the "colour.sd_blackbody" definition warning.


-   colour/characterisation/correction.py

    -   Line 459 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1236 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 422 : # TODO: Check for interpolator compatibility.
    -   Line 482 : # TODO: Check for extrapolator compatibility.


-   colour/hints/__init__.py

    -   Line 137 : # TODO: Revisit to use Protocol.


-   colour/algebra/tests/test_interpolation.py

    -   Line 1168 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers - `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
