Colour - TODO
=============

TODO
----

- colour (22 items in 15 files)

    - appearance (7 items in 5 files)

        - ciecam02.py

            - (257, 7) # TODO: Compute hue composition.

        - hunt.py

            - (384, 7) # TODO: Implement hue quadrature & composition computation.
            - (415, 7) # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.

        - llab.py

            - (297, 7) # TODO: Implement hue composition computation.

        - nayatani95.py

            - (244, 7) # TODO: Implement hue quadrature & composition computation.
            - (256, 7) # TODO: Investigate components usage.

        - rlab.py

            - (235, 7) # TODO: Implement hue composition computation.

    - colorimetry (4 items in 3 files)

        - tests (2 items in 1 file)

            - tests_spectrum.py

                - (2352, 11) # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version.
                - (2834, 11) # TODO: Remove statement whenever we make "Scipy" 0.19.0 the minimum version.

        - spectrum.py

            - (1920, 11) # TODO: Provide support for fractional interval like 0.1, etc...

        - tristimulus.py

            - (672, 11) # TODO: Investigate code vectorisation.

    - models (4 items in 3 files)

        - rgb (4 items in 3 files)

            - tests (1 item in 1 file)

                - tests_derivation.py

                    - (275, 11) # TODO: Simplify that monster.

            - derivation.py

                - (215, 7) # TODO: Investigate if we return an ndarray here with primaries and

            - rgb_colourspace.py

                - (515, 11) # TODO: Revisit for potential behaviour / type checking.
                - (542, 11) # TODO: Revisit for potential behaviour / type checking.

    - notation (5 items in 2 files)

        - tests (3 items in 1 file)

            - tests_munsell.py

                - (67, 3) # TODO: Investigate if tests can be simplified by using a common valid set of specifications.
                - (4399, 11) # TODO: This test is covered by the previous class, do we need a dedicated one?
                - (4441, 11) # TODO: This test is covered by the previous class, do we need a dedicated one?

        - munsell.py

            - (802, 11) # TODO: Consider refactoring implementation.
            - (1129, 11) # TODO: Should raise KeyError, need to check the tests.

    - plotting (1 item in 1 file)

        - colorimetry.py

            - (599, 11) # TODO: Handle condition statement with metadata capabilities.

    - volume (1 item in 1 file)

        - rgb.py

            - (308, 7) # TODO: Investigate for generator yielding directly a ndarray.

About
-----

| **Colour** by Colour Developers - 2013-2017
| Copyright © 2013-2017 – Colour Developers – `colour-science@googlegroups.com <colour-science@googlegroups.com>`_
| This software is released under terms of New BSD License: http://opensource.org/licenses/BSD-3-Clause
| `http://github.com/colour-science/colour <http://github.com/colour-science/colour>`_
