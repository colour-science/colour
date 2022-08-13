from __future__ import annotations

from colour.utilities import CanonicalMapping
from .aces import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACE_ACESCC,
    RGB_COLOURSPACE_ACESCCT,
    RGB_COLOURSPACE_ACESCG,
    RGB_COLOURSPACE_ACESPROXY,
)
from .adobe_rgb_1998 import RGB_COLOURSPACE_ADOBE_RGB1998
from .adobe_wide_gamut_rgb import RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB
from .apple_rgb import RGB_COLOURSPACE_APPLE_RGB
from .arri_alexa_wide_gamut import RGB_COLOURSPACE_ALEXA_WIDE_GAMUT
from .best_rgb import RGB_COLOURSPACE_BEST_RGB
from .beta_rgb import RGB_COLOURSPACE_BETA_RGB
from .blackmagic_design import RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT
from .cie_rgb import RGB_COLOURSPACE_CIE_RGB
from .canon_cinema_gamut import RGB_COLOURSPACE_CINEMA_GAMUT
from .color_match_rgb import RGB_COLOURSPACE_COLOR_MATCH_RGB
from .davinci_wide_gamut import RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT
from .dcdm_xyz import RGB_COLOURSPACE_DCDM_XYZ
from .dci_p3 import RGB_COLOURSPACE_DCI_P3, RGB_COLOURSPACE_DCI_P3_P
from .display_p3 import RGB_COLOURSPACE_DISPLAY_P3
from .dji_dgamut import RGB_COLOURSPACE_DJI_D_GAMUT
from .don_rgb_4 import RGB_COLOURSPACE_DON_RGB_4
from .ebu_3213_e import RGB_COLOURSPACE_EBU_3213_E
from .eci_rgb_v2 import RGB_COLOURSPACE_ECI_RGB_V2
from .ekta_space_ps5 import RGB_COLOURSPACE_EKTA_SPACE_PS_5
from .fujifilm_f_gamut import RGB_COLOURSPACE_F_GAMUT
from .filmlight_egamut import RGB_COLOURSPACE_FILMLIGHT_E_GAMUT
from .gopro import RGB_COLOURSPACE_PROTUNE_NATIVE
from .itur_bt_470 import RGB_COLOURSPACE_BT470_525, RGB_COLOURSPACE_BT470_625
from .itur_bt_709 import RGB_COLOURSPACE_BT709
from .itur_bt_2020 import RGB_COLOURSPACE_BT2020
from .itut_h_273 import (
    RGB_COLOURSPACE_H273_GENERIC_FILM,
    RGB_COLOURSPACE_H273_22_UNSPECIFIED,
)
from .max_rgb import RGB_COLOURSPACE_MAX_RGB
from .nikon_n_gamut import RGB_COLOURSPACE_N_GAMUT
from .p3_d65 import RGB_COLOURSPACE_P3_D65
from .pal_secam import RGB_COLOURSPACE_PAL_SECAM
from .red import (
    RGB_COLOURSPACE_RED_COLOR,
    RGB_COLOURSPACE_RED_COLOR_2,
    RGB_COLOURSPACE_RED_COLOR_3,
    RGB_COLOURSPACE_RED_COLOR_4,
    RGB_COLOURSPACE_DRAGON_COLOR,
    RGB_COLOURSPACE_DRAGON_COLOR_2,
    RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB,
)
from .rimm_romm_rgb import (
    RGB_COLOURSPACE_ROMM_RGB,
    RGB_COLOURSPACE_RIMM_RGB,
    RGB_COLOURSPACE_ERIMM_RGB,
    RGB_COLOURSPACE_PROPHOTO_RGB,
)
from .russell_rgb import RGB_COLOURSPACE_RUSSELL_RGB
from .sharp import RGB_COLOURSPACE_SHARP_RGB
from .smpte_240m import RGB_COLOURSPACE_SMPTE_240M
from .smpte_c import RGB_COLOURSPACE_SMPTE_C
from .ntsc import RGB_COLOURSPACE_NTSC1953, RGB_COLOURSPACE_NTSC1987
from .sony import (
    RGB_COLOURSPACE_S_GAMUT,
    RGB_COLOURSPACE_S_GAMUT3,
    RGB_COLOURSPACE_S_GAMUT3_CINE,
    RGB_COLOURSPACE_VENICE_S_GAMUT3,
    RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE,
)
from .srgb import RGB_COLOURSPACE_sRGB
from .panasonic_v_gamut import RGB_COLOURSPACE_V_GAMUT
from .xtreme_rgb import RGB_COLOURSPACE_XTREME_RGB

from colour.models.rgb import RGB_Colourspace

RGB_COLOURSPACES: CanonicalMapping = CanonicalMapping(
    dict(
        sorted(
            (colourspace.name, colourspace)
            for colourspace in locals().values()
            if isinstance(colourspace, RGB_Colourspace)
        )
    )
)
RGB_COLOURSPACES.__doc__ = """
Aggregated *RGB* colourspaces.

Aliases:

-   'aces': RGB_COLOURSPACE_ACES2065_1.name
-   'adobe1998': RGB_COLOURSPACE_ADOBE_RGB1998.name
-   'prophoto': RGB_COLOURSPACE_PROPHOTO_RGB.name
"""

RGB_COLOURSPACES["aces"] = RGB_COLOURSPACES[RGB_COLOURSPACE_ACES2065_1.name]
RGB_COLOURSPACES["adobe1998"] = RGB_COLOURSPACES[
    RGB_COLOURSPACE_ADOBE_RGB1998.name
]
RGB_COLOURSPACES["prophoto"] = RGB_COLOURSPACES[
    RGB_COLOURSPACE_PROPHOTO_RGB.name
]
# yapf: enable

__all__ = [
    "RGB_COLOURSPACES",
]
__all__ += [
    "RGB_COLOURSPACE_ACES2065_1",
    "RGB_COLOURSPACE_ACESCC",
    "RGB_COLOURSPACE_ACESCCT",
    "RGB_COLOURSPACE_ACESPROXY",
    "RGB_COLOURSPACE_ACESCG",
    "RGB_COLOURSPACE_ADOBE_RGB1998",
    "RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_ALEXA_WIDE_GAMUT",
    "RGB_COLOURSPACE_APPLE_RGB",
    "RGB_COLOURSPACE_BEST_RGB",
    "RGB_COLOURSPACE_BETA_RGB",
    "RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT",
    "RGB_COLOURSPACE_BT470_525",
    "RGB_COLOURSPACE_BT470_625",
    "RGB_COLOURSPACE_BT709",
    "RGB_COLOURSPACE_BT2020",
    "RGB_COLOURSPACE_H273_GENERIC_FILM",
    "RGB_COLOURSPACE_H273_22_UNSPECIFIED",
    "RGB_COLOURSPACE_CIE_RGB",
    "RGB_COLOURSPACE_CINEMA_GAMUT",
    "RGB_COLOURSPACE_COLOR_MATCH_RGB",
    "RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT",
    "RGB_COLOURSPACE_DCDM_XYZ",
    "RGB_COLOURSPACE_DCI_P3",
    "RGB_COLOURSPACE_DCI_P3_P",
    "RGB_COLOURSPACE_DISPLAY_P3",
    "RGB_COLOURSPACE_DJI_D_GAMUT",
    "RGB_COLOURSPACE_EBU_3213_E",
    "RGB_COLOURSPACE_DON_RGB_4",
    "RGB_COLOURSPACE_ECI_RGB_V2",
    "RGB_COLOURSPACE_EKTA_SPACE_PS_5",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT",
    "RGB_COLOURSPACE_PROTUNE_NATIVE",
    "RGB_COLOURSPACE_MAX_RGB",
    "RGB_COLOURSPACE_N_GAMUT",
    "RGB_COLOURSPACE_P3_D65",
    "RGB_COLOURSPACE_PAL_SECAM",
    "RGB_COLOURSPACE_RED_COLOR",
    "RGB_COLOURSPACE_RED_COLOR_2",
    "RGB_COLOURSPACE_RED_COLOR_3",
    "RGB_COLOURSPACE_RED_COLOR_4",
    "RGB_COLOURSPACE_DRAGON_COLOR",
    "RGB_COLOURSPACE_DRAGON_COLOR_2",
    "RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_ROMM_RGB",
    "RGB_COLOURSPACE_RIMM_RGB",
    "RGB_COLOURSPACE_ERIMM_RGB",
    "RGB_COLOURSPACE_PROPHOTO_RGB",
    "RGB_COLOURSPACE_RUSSELL_RGB",
    "RGB_COLOURSPACE_SHARP_RGB",
    "RGB_COLOURSPACE_SMPTE_240M",
    "RGB_COLOURSPACE_SMPTE_C",
    "RGB_COLOURSPACE_NTSC1953",
    "RGB_COLOURSPACE_NTSC1987",
    "RGB_COLOURSPACE_S_GAMUT",
    "RGB_COLOURSPACE_S_GAMUT3",
    "RGB_COLOURSPACE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_sRGB",
    "RGB_COLOURSPACE_V_GAMUT",
    "RGB_COLOURSPACE_XTREME_RGB",
    "RGB_COLOURSPACE_F_GAMUT",
]
