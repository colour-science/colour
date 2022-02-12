"""Showcases RGB colourspace volume computations."""

import colour
from colour.utilities import message_box

# NOTE: Because the MonteCarlo methods use multiprocessing, it is recommended
# to wrap the execution in a definition or a *__main__* block.
if __name__ == "__main__":
    message_box("RGB Colourspace Volume Computations")

    message_box('Computing the "ProPhoto RGB" RGB colourspace limits.')
    limits = colour.RGB_colourspace_limits(
        colour.RGB_COLOURSPACES["ProPhoto RGB"]
    )
    print(limits)

    print("\n")

    samples = 10e4
    message_box(
        f'Computing the"ProPhoto RGB" RGB colourspace volume using {samples} '
        f"samples."
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
        f'Computing "ProPhoto RGB" RGB colourspace coverage of '
        f'"Pointer\'s Gamut" using {samples} samples.'
    )
    print(
        colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
            colour.RGB_COLOURSPACES["ProPhoto RGB"], samples=samples
        )
    )
