Input and Output
================

Image Data
----------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    READ_IMAGE_METHODS
    read_image
    WRITE_IMAGE_METHODS
    write_image

**Ancillary Objects**

``colour.io``

.. currentmodule:: colour.io

.. autosummary::
    :toctree: generated/

    ImageAttribute_Specification
    convert_bit_depth
    read_image_OpenImageIO
    write_image_OpenImageIO
    read_image_Imageio
    write_image_Imageio
    as_3_channels_image

OpenColorIO Processing
----------------------

``colour.io``

.. currentmodule:: colour.io

.. autosummary::
    :toctree: generated/

    process_image_OpenColorIO

Look Up Table (LUT) Data
------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    LUT1D
    LUT3x1D
    LUT3D
    LUTOperatorMatrix
    LUTSequence

.. autosummary::
    :toctree: generated/

    read_LUT
    write_LUT

**Ancillary Objects**

``colour.io``

.. currentmodule:: colour.io

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AbstractLUTSequenceOperator

.. autosummary::
    :toctree: generated/

    LUT_to_LUT
    read_LUT_Cinespace
    write_LUT_Cinespace
    read_LUT_IridasCube
    write_LUT_IridasCube
    read_LUT_SonySPI1D
    write_LUT_SonySPI1D
    read_LUT_SonySPI3D
    write_LUT_SonySPI3D

CSV Tabular Data
----------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    read_sds_from_csv_file
    read_spectral_data_from_csv_file
    write_sds_to_csv_file

IES TM-27-14 Data
-----------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SpectralDistribution_IESTM2714


UPRTek and Sekonic Spectral Data
--------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SpectralDistribution_UPRTek
    SpectralDistribution_Sekonic

X-Rite Data
-----------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    read_sds_from_xrite_file
