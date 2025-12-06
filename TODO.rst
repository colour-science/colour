Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 931 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/tristimulus_values.py

    -   Line 1076 : # TODO: Investigate code vectorisation.


-   colour/appearance/ciecam02.py

    -   Line 358 : # TODO: Compute hue composition.


-   colour/appearance/ciecam16.py

    -   Line 315 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 322 : # TODO: Compute hue composition.


-   colour/appearance/hellwig2022.py

    -   Line 354 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 487 : # TODO: Implement hue quadrature & composition computation.
    -   Line 518 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 285 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 313 : # TODO: Implement hue quadrature & composition computation.
    -   Line 324 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 368 : # TODO: Implement hue composition computation.


-   colour/io/fichet2021.py

    -   Line 692 : # TODO: Implement support for integration of bi-spectral component.
    -   Line 699 : # TODO: Implement support for re-binning component with non-uniform interval.


-   colour/io/ctl.py

    -   Line 65 : # TODO: Reinstate coverage when "ctlrender" is trivially available cross-platform.


-   colour/io/tests/test_ocio.py

    -   Line 37 : # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/tests/test_ctl.py

    -   Line 39 : # TODO: Reinstate coverage when "ctlrender" is tivially available cross-platform.


-   colour/io/tests/test_image.py

    -   Line 335 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/models/rgb/derivation.py

    -   Line 228 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/tests/test_rgb_colourspace.py

    -   Line 342 : # TODO: Remove tests when dropping deprecated signature support.
    -   Line 542 : # TODO: Remove tests when dropping deprecated signature support.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 329 : # TODO: Simplify that monster.


-   colour/utilities/verbose.py

    -   Line 819 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library. NOTE: A few clauses are not reached and a few packages are not available during continuous integration and are thus ignored for coverage.


-   colour/utilities/network.py

    -   Line 603 : # TODO: Consider using an ordered set instead of a dict.
    -   Line 1089 : # TODO: Consider using ordered set.
    -   Line 1095 : # TODO: Consider using ordered set.
    -   Line 1958 : # TODO: Implement solid control flow based processing using a stack.


-   colour/utilities/array.py

    -   Line 606 : # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is addressed.
    -   Line 825 : # TODO: Revisit when Numpy types are well established.
    -   Line 863 : # TODO: Revisit when Numpy types are well established.
    -   Line 944 : # TODO: Investigate behaviour on Windows.
    -   Line 1008 : # TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]" when Python 3.7 is dropped.


-   colour/plotting/models.py

    -   Line 1950 : # TODO: Filter appropriate colour models. NOTE: "dtype=object" is required for ragged array support in "Numpy" 1.24.0.


-   colour/plotting/graph.py

    -   Line 88 : # TODO: Investigate API to trigger the conversion graph build.


-   colour/plotting/common.py

    -   Line 877 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 993 : # TODO: Consider using "MutableMapping" here.


-   colour/characterisation/correction.py

    -   Line 468 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1260 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 417 : # TODO: Check for interpolator compatibility.
    -   Line 475 : # TODO: Check for extrapolator compatibility.


-   colour/hints/__init__.py

    -   Line 150 : # TODO: Revisit to use Protocol.


-   colour/algebra/tests/test_interpolation.py

    -   Line 1176 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers - `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
