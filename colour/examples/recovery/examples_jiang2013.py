"""
Showcases camera *RGB* sensitivities recovery computations using
*Jiang et al. (2013)* method.
"""

import colour
from colour.utilities import message_box

message_box('"Jiang et al. (2013)" -  Camera Sensitivities Recovery')

shape = colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017

illuminant = colour.SDS_ILLUMINANTS["D65"]
sensitivities = colour.MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]
reflectances = colour.colorimetry.sds_and_msds_to_msds(
    [
        sd.copy().align(shape)
        for sd in colour.SDS_COLOURCHECKERS["BabelColor Average"].values()
    ]
)
RGB = colour.msds_to_XYZ(
    reflectances,
    method="Integration",
    cmfs=sensitivities,
    illuminant=illuminant,
    k=1,
    shape=colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
)
msds_camera_sensitivities = colour.recovery.RGB_to_msds_camera_sensitivities_Jiang2013(
    RGB,
    illuminant,
    reflectances,
    colour.recovery.BASIS_FUNCTIONS_DYER2017,
    colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
)

message_box(
    f'Recovering camera *RGB* sensitivities using "Jiang et al. (2013)" method:'
    f"\n\n{msds_camera_sensitivities}"
)
