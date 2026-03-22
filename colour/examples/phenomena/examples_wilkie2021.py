"""
Demonstrate *Wilkie et al. (2021)* sky model computations.

This module demonstrates *Prague Sky Model* sky radiance, sun radiance
and transmittance computations as presented in *Wilkie et al. (2021)*.

Note: This example requires a *Prague Sky Model* dataset file. Sync it with
``colour_datasets.load("19140728")``.
"""

import os

import numpy as np

from colour.phenomena.sky.wilkie2021 import (
    PATH_PRAGUE_SKY_MODEL_DATASET_GROUND,
    SkyDataset_Wilkie2021,
    compute_sky_parameters_Wilkie2021,
    sky_radiance_Wilkie2021,
    sky_transmittance_Wilkie2021,
    sun_radiance_Wilkie2021,
)
from colour.utilities import message_box

message_box("Prague Sky Model Computations")

if not os.path.isfile(PATH_PRAGUE_SKY_MODEL_DATASET_GROUND):
    print(
        "Prague Sky Model dataset not found.\n"
        'Sync it with: colour_datasets.load("19140728")'
    )
else:
    message_box("Loading Prague Sky Model dataset...")
    dataset = SkyDataset_Wilkie2021(PATH_PRAGUE_SKY_MODEL_DATASET_GROUND)
    print(
        f"Channels: {dataset.channels}, "
        f"Range: {dataset.channel_start}-"
        f"{dataset.channel_start + dataset.channels * dataset.channel_width} nm"
    )

    print("\n")

    message_box(
        "Computing sky radiance for an observer at ground level,\n"
        "looking at zenith, with the sun at 30 degrees elevation."
    )
    params = compute_sky_parameters_Wilkie2021(
        view_point=np.array([0, 0, 0.0]),
        view_direction=np.array([0, 0, 1.0]),
        sun_elevation=np.radians(30),
        sun_azimuth=0.0,
        visibility=50.0,
        albedo=0.5,
    )
    wavelength = np.array([380.0, 460.0, 540.0, 620.0, 700.0])
    radiance = sky_radiance_Wilkie2021(dataset, params, wavelength)
    print(f"Wavelengths: {wavelength}")
    print(f"Sky radiance: {radiance}")

    print("\n")

    message_box("Computing transmittance to infinity at the same configuration.")
    transmittance = sky_transmittance_Wilkie2021(dataset, params, wavelength, np.inf)
    print(f"Transmittance: {transmittance}")

    print("\n")

    message_box("Computing sun radiance (looking directly at the sun).")
    sun_dir = np.array(
        [
            np.cos(0.0) * np.cos(np.radians(30)),
            np.sin(0.0) * np.cos(np.radians(30)),
            np.sin(np.radians(30)),
        ]
    )
    params_sun = compute_sky_parameters_Wilkie2021(
        view_point=np.array([0, 0, 0.0]),
        view_direction=sun_dir,
        sun_elevation=np.radians(30),
        sun_azimuth=0.0,
        visibility=50.0,
        albedo=0.5,
    )
    sun_radiance = sun_radiance_Wilkie2021(dataset, params_sun, wavelength)
    print(f"Sun radiance: {sun_radiance}")
