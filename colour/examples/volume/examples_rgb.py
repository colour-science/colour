"""
Demonstrate RGB colourspace volume computations.

This module demonstrates the computation of RGB colourspace limits, volumes,
and gamut coverage using Monte Carlo methods for colourspace analysis.
"""

import colour
from colour.utilities import message_box

# NOTE: Because the MonteCarlo methods use multiprocessing, it is recommended
# to wrap the execution in a definition or a *__main__* block.
if __name__ == "__main__":
    message_box("RGB Colourspace Volume Computations")

    message_box('Compute the "ProPhoto RGB" RGB colourspace limits.')
    limits = colour.RGB_colourspace_limits(colour.RGB_COLOURSPACES["ProPhoto RGB"])
    print(limits)

    print("\n")

    samples = int(10e4)
    message_box(
        f'Compute the "ProPhoto RGB" RGB colourspace volume using {samples} samples.'
    )
    print(
        colour.RGB_colourspace_volume_MonteCarlo(
            colour.RGB_COLOURSPACES["ProPhoto RGB"],
            samples=samples,
            limits=limits * 1.1,
        )
    )

    print("\n")

    message_box(
        f'Compute "ProPhoto RGB" RGB colourspace coverage of '
        f'"Pointer\'s Gamut" using {samples} samples.'
    )
    print(
        colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
            colour.RGB_COLOURSPACES["ProPhoto RGB"], samples=samples
        )
    )
