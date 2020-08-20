from colour.colorimetry import SDS_ILLUMINANTS
from colour.quality import colour_fidelity_index_TM_30_18

from _elements import *


if __name__ == '__main__':
    lamp = SDS_ILLUMINANTS['FL2']
    spec = colour_fidelity_index_TM_30_18(lamp, True)

    plt.subplot(4, 2, 1)
    plot_spectra_TM_30_18(spec)
    plt.subplot(4, 2, (3, 5))
    plot_color_vector_graphic(spec)
    plt.subplot(4, 2, 2)
    plot_local_chroma_shifts(spec)
    plt.subplot(4, 2, 4)
    plot_local_hue_shifts(spec)
    plt.subplot(4, 2, 6)
    plot_local_color_fidelities(spec)
    plt.subplot(4, 2, (7, 8))
    plot_colour_fidelity_indexes(spec)
    plt.show()