"""
Demonstrate thin film interference plotting.

This module provides examples of plotting thin film optical properties
using the *Transfer Matrix Method* (TMM) implementation, including single
and multilayer films, anti-reflection coatings, and Bragg reflectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from colour.colorimetry import SPECTRAL_SHAPE_DEFAULT
from colour.phenomena.interference import light_water_refractive_index_Schiebener1990
from colour.plotting import colour_style
from colour.plotting.phenomena import (
    plot_multi_layer_stack,
    plot_multi_layer_thin_film,
    plot_single_layer_thin_film,
    plot_thin_film_comparison,
    plot_thin_film_iridescence,
    plot_thin_film_reflectance_map,
    plot_thin_film_spectrum,
)
from colour.utilities import message_box

message_box("Thin Film Interference Plots")

colour_style()

# ==============================================================================
# Single Layer Thin Film - Basic Plot
# ==============================================================================

message_box(
    "Single Layer Thin Film Reflectance\n\n"
    "Plot reflectance vs wavelength for a thin glass film:\n"
    "  - Film: Glass (n=1.46)\n"
    "  - Substrate: Glass (n=1.52)\n"
    "  - Thickness: 100 nm\n"
    "  - Normal incidence"
)

plot_single_layer_thin_film(
    n=[1.0, 1.46, 1.52],  # [air, film, substrate]
    t=100,
    theta=0,
    polarisation="Both",
    method="Reflectance",
)

print("\n")

# ==============================================================================
# Anti-Reflection Coating
# ==============================================================================

message_box(
    "Anti-Reflection (AR) Coating\n\n"
    "Quarter-wave MgF₂ coating on glass:\n"
    "  - Film: MgF₂ (n=1.38)\n"
    "  - Substrate: Glass (n=1.52)\n"
    "  - Thickness: λ/(4n) at 550nm ≈ 100 nm\n"
    "  - Shows minimum reflectance at design wavelength"
)

# Quarter-wave thickness for 550nm
wavelength_design = 550
n_mgf2 = 1.38
thickness_ar = wavelength_design / (4 * n_mgf2)

plot_single_layer_thin_film(
    n=[1.0, n_mgf2, 1.52],  # [air, MgF2, glass]
    t=thickness_ar,
    theta=0,
    polarisation="Both",
    method="Both",
)

print("\n")

# ==============================================================================
# Thin Film Comparison
# ==============================================================================

message_box(
    "Thin Film Comparison\\n\\n"
    "Compare different thin film materials:\\n"
    "  - MgF₂ (n=1.38): 100 nm quarter-wave AR coating\\n"
    "  - TiO₂ (n=2.4): 60 nm high-index coating\\n"
    "  - SiO₂ (n=1.46): 120 nm mid-index coating\\n"
    "  - Shows how refractive index affects reflectance"
)

configurations = [
    {
        "type": "single",
        "n_film": 1.38,  # MgF2
        "t": 100,
        "n_substrate": 1.52,
        "label": "MgF₂ 100nm",
    },
    {
        "type": "single",
        "n_film": 2.4,  # TiO2
        "t": 60,
        "n_substrate": 1.52,
        "label": "TiO₂ 60nm",
    },
    {
        "type": "single",
        "n_film": 1.46,  # SiO2
        "t": 120,
        "n_substrate": 1.52,
        "label": "SiO₂ 120nm",
    },
]

plot_thin_film_comparison(configurations, polarisation="Both")

print("\n")

# ==============================================================================
# Multilayer Stack - Bragg Reflector
# ==============================================================================

message_box(
    "Bragg Reflector (Distributed Bragg Reflector)\n\n"
    "Multilayer quarter-wave stack:\n"
    "  - 5 pairs of high/low index layers\n"
    "  - High index: TiO₂ (n=2.0)\n"
    "  - Low index: SiO₂ (n=1.46)\n"
    "  - Design wavelength: 600 nm\n"
    "  - Shows high reflectance peak"
)

# Design parameters
n_high = 2.0  # TiO2
n_low = 1.46  # SiO2
wl_design = 600

# Quarter-wave thicknesses
d_high = wl_design / (4 * n_high)
d_low = wl_design / (4 * n_low)

# Create 5 pairs (10 layers)
layer_indices = [n_high, n_low] * 5
layer_thicknesses = [d_high, d_low] * 5

# Build unified n array: [air, layers..., substrate]
n_stack = [1.0, *layer_indices, 1.52]

plot_multi_layer_thin_film(
    n=n_stack,
    t=layer_thicknesses,
    theta=0,
    polarisation="Both",
    method="Both",
)

print("\n")

# ==============================================================================
# Anti-Reflection Multilayer
# ==============================================================================

message_box(
    "Broadband AR Coating\n\n"
    "Two-layer anti-reflection coating:\n"
    "  - Layer 1: n=1.38, d=100 nm\n"
    "  - Layer 2: n=2.0, d=50 nm\n"
    "  - Broader AR band than single layer"
)

plot_multi_layer_thin_film(
    n=[1.0, 1.38, 2.0, 1.52],  # [air, layer1, layer2, substrate]
    t=[100, 50],
    theta=0,
    polarisation="Both",
    method="Transmittance",
)

print("\n")

# ==============================================================================
# Interference Spectrum (Simple Model)
# ==============================================================================

message_box(
    "Thin Film Interference Spectrum (Simple Model)\n\n"
    "Soap film interference colours:\n"
    "  - Film: Water/soap (n=1.33)\n"
    "  - Thickness: 300 nm\n"
    "  - Shows characteristic oscillations\n"
    "  - Uses simple interference formula"
)

plot_thin_film_spectrum(
    n=[1.0, 1.33, 1.0],  # [air, film, air]
    t=300,
    theta=0,
)

print("\n")

# ==============================================================================
# Thicker Film - More Fringes
# ==============================================================================

message_box(
    "Thicker Film with More Interference Fringes\n\n"
    "  - Thickness: 800 nm\n"
    "  - Shows more oscillations\n"
    "  - Characteristic of thicker films"
)

plot_thin_film_spectrum(
    n=[1.0, 1.5, 1.0],  # [air, film, air]
    t=800,
    theta=0,
)

print("\n")

# ==============================================================================
# Colour vs Thickness - Soap Film Colours
# ==============================================================================

message_box(
    "Thin Film Colours vs Thickness\n\n"
    "Shows how colours change with film thickness:\n"
    "  - Like soap bubbles or oil slicks\n"
    "  - Thickness range: 0-1000 nm\n"
    "  - Uses wavelength-dependent water refractive index\n"
    "  - Creates iridescent colour strip"
)

# Build dispersive n array for water
wavelengths = SPECTRAL_SHAPE_DEFAULT.wavelengths
n_water = light_water_refractive_index_Schiebener1990(
    wavelengths, temperature=294, density=1000
)
# Stack: [air, water(dispersive), air]
n_dispersive = np.array([np.ones_like(wavelengths), n_water, np.ones_like(wavelengths)])

plot_thin_film_iridescence(
    n=n_dispersive,
    t=None,  # Default 0-1000 nm
    theta=0,
)

print("\n")

# ==============================================================================
# Colour vs Thickness - Custom Range
# ==============================================================================

message_box(
    "Thin Film Colours - Focused Range\n\n"
    "Zoomed in on first few interference orders:\n"
    "  - Thickness range: 0-500 nm\n"
    "  - Shows first Newton's rings colours\n"
    "  - Black, blue, yellow, red sequence"
)

# Build dispersive n array for water (same as above)
n_water_focused = light_water_refractive_index_Schiebener1990(
    wavelengths, temperature=294, density=1000
)
n_dispersive_focused = np.array(
    [np.ones_like(wavelengths), n_water_focused, np.ones_like(wavelengths)]
)

plot_thin_film_iridescence(
    n=n_dispersive_focused,
    t=np.arange(0, 500, 1),  # 0-500 nm
    theta=0,
)

print("\n")

# ==============================================================================
# Reflectance Map - Reflectance vs Wavelength and Thickness
# ==============================================================================

message_box(
    "Reflectance Map: Reflectance vs Wavelength and Thickness\n\n"
    "2D visualization of thin film interference:\n"
    "  - X-axis: Wavelength (380-780 nm)\n"
    "  - Y-axis: Film thickness (0-1000 nm)\n"
    "  - Colour: Reflectance intensity\n"
    "  - Shows interference fringes as horizontal bands\n"
    "  - Soap film (n=1.33) at normal incidence"
)

plot_thin_film_reflectance_map(
    n=[1.0, 1.33, 1.0],  # [air, soap, air]
    t=None,  # Default 0-1000 nm with 250 points
    theta=0,  # Normal incidence
    polarisation="Average",
    method="Thickness",
)

print("\n")

# ==============================================================================
# Reflectance Map - Angle Mode (Reflectance vs Wavelength and Angle)
# ==============================================================================

message_box(
    "Reflectance Map: Reflectance vs Wavelength and Angle\n\n"
    "2D visualization showing angular dependence:\n"
    "  - X-axis: Wavelength (380-780 nm)\n"
    "  - Y-axis: Incident angle (0-90°)\n"
    "  - Colour: Reflectance intensity\n"
    "  - Shows how reflectance varies with angle\n"
    "  - Soap film (n=1.33) at 300 nm thickness"
)

plot_thin_film_reflectance_map(
    n=[1.0, 1.33, 1.0],  # [air, soap, air]
    t=300,  # Fixed thickness in nm
    theta=np.linspace(0, 90, 250),  # Array of angles
    polarisation="Average",
    method="Angle",  # Explicit angle mode
)

print("\n")

# ==============================================================================
# Combined Reflectance Map and Multi-layer Stack Visualization
# ==============================================================================

message_box(
    "Combined Reflectance Map and Multi-layer Stack Visualization\n\n"
    "Reproduces the optimized 4-layer thin film stack:\n"
    "  - Left: Reflectance map (wavelength vs angle)\n"
    "  - Right: Layer stack diagram\n"
    "  - Layer 1: n=2.7538, d=100 nm (TiO₂ rutile)\n"
    "  - Layer 2: n=3.0000, d=180 nm (Si/GaAs)\n"
    "  - Layer 3: n=2.6666, d=80 nm (TiO₂)\n"
    "  - Layer 4: n=2.1595, d=200 nm (ZrO₂)\n"
    "  - Total thickness: 560 nm"
)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

refractive_indices = [1.0, 2.75, 3.0000, 2.6666, 2.15, 1.50]
thicknesses = [100, 180, 80, 200]  # nm

# Left subplot: Reflectance map
plot_thin_film_reflectance_map(
    n=refractive_indices,
    t=thicknesses,
    theta=np.linspace(0, 90, 300),
    polarisation="Average",
    method="Angle",
    wavelength=np.linspace(300, 1200, 400),
    axes=ax1,
    show=False,
)
ax1.set_title("Reflectance Map (4-Layer Stack)")

# Right subplot: Multi-layer stack diagram
# Build configuration for stack visualization
configurations = [
    {"t": thicknesses[0], "n": refractive_indices[1], "color": "#FF69B4"},  # Pink
    {"t": thicknesses[1], "n": refractive_indices[2], "color": "#4169E1"},  # Blue
    {"t": thicknesses[2], "n": refractive_indices[3], "color": "#FF8C00"},  # Orange
    {"t": thicknesses[3], "n": refractive_indices[4], "color": "#9370DB"},  # Purple
]

plot_multi_layer_stack(
    configurations=configurations,
    n_incident=refractive_indices[0],
    n_substrate=refractive_indices[-1],
    theta=45,
    axes=ax2,
    show=False,
)
ax2.set_title("Multi-layer Stack Structure")

plt.tight_layout()
plt.show()

print("\n")
