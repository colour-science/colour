"""
Demonstrate *Transfer Matrix Method* (TMM) for Thin Film Interference.

This module demonstrates thin film optical calculations using the Transfer
Matrix Method, including reflectance and transmittance computations for
single-layer and multilayer thin films at various wavelengths and incident
angles.

Common Refractive Indices (at 589 nm, 20°C)
-------------------------------------------
- Air/Vacuum: n = 1.0
- Water: n = 1.33
- Fused Silica (SiO₂): n = 1.46
- Typical Glass: n = 1.5
- Crown Glass (BK7): n = 1.52
- Flint Glass: n = 1.6-1.9
- Magnesium Fluoride (MgF₂): n = 1.38 (common AR coating)
- Titanium Dioxide (TiO₂): n = 2.0-2.7 (high-index coating)
"""

import numpy as np

import colour
from colour.colorimetry import SPECTRAL_SHAPE_DEFAULT
from colour.utilities import message_box

message_box('"Transfer Matrix Method" (TMM) Thin Film Calculations')

# ==============================================================================
# Single Layer Thin Film - Basic Reflectance
# ==============================================================================

message_box(
    "Single Layer Thin Film Reflectance\n\n"
    "Calculate reflectance of a single dielectric layer:\n"
    "  - Film: Glass (n=1.5)\n"
    "  - Thickness: 250 nm\n"
    "  - Wavelength: 550 nm\n"
    "  - Normal incidence (0°)"
)

R, T = colour.phenomena.thin_film_tmm(
    n=[1.0, 1.5, 1.0],  # air | film | air
    t=250,
    wavelength=550,
    theta=0,
)

print(f"Reflectance [R_s, R_p]: {R[0, 0, 0]}")
print(
    f"Average reflectance: {np.mean(R[0, 0, 0]):.4f} ({np.mean(R[0, 0, 0]) * 100:.2f}%)"
)

print("\n")

# ==============================================================================
# Energy Conservation
# ==============================================================================

message_box(
    "Energy Conservation: R + T = 1\n\n"
    "Verify that reflectance + transmittance = 1 for non-absorbing films"
)

# Unified API returns both R and T together
# (Already computed in previous section, just using those values)

print(f"Reflectance R: {R[0, 0, 0]}")
print(f"Transmittance T: {T[0, 0, 0]}")
print(f"R + T: {R[0, 0, 0] + T[0, 0, 0]}")
print(f"Energy conserved: {np.allclose(R + T, 1.0)}")

print("\n")

# ==============================================================================
# Wavelength-Dependent Colors
# ==============================================================================

message_box(
    "Thin Film Interference Colors\n\n"
    "Calculate reflectance across the visible spectrum:\n"
    "  - Film: Soap film (n=1.33)\n"
    "  - Thickness: 300 nm"
)

wavelengths = SPECTRAL_SHAPE_DEFAULT.wavelengths
# Soap film in air: uses defaults n_substrate=1.0 (air), n_incident=1.0 (air), theta_i=0
R_spectrum, T_spectrum = colour.phenomena.thin_film_tmm(
    n=[1.0, 1.33, 1.0], t=300, wavelength=wavelengths
)

# Show a few sample values at 400, 500, 600, 700 nm
print("Wavelength (nm) | Reflectance (s-pol)")
print(f"{'─' * 40}")
for wl_sample in [400, 500, 600, 700]:
    idx = int(wl_sample - 360)  # SPECTRAL_SHAPE_DEFAULT starts at 360 nm
    r = R_spectrum[idx, 0, 0, 0]  # Shape is now (W, A, T, 2) - Spectroscopy Convention
    print(f"  {wl_sample:6.0f}        | {r:.6f} ({r * 100:5.2f}%)")

print(
    f"\nReflectance varies from {R_spectrum[:, 0, 0, 0].min():.4f} "
    f"to {R_spectrum[:, 0, 0, 0].max():.4f}"
)
print("This variation creates the colorful appearance of thin films!")

print("\n")

# ==============================================================================
# Anti-Reflection Coating
# ==============================================================================

message_box(
    "Anti-Reflection (AR) Coating Design\n\n"
    "Quarter-wave AR coating:\n"
    "  - Substrate: Glass (n=1.5)\n"
    "  - Coating: MgF₂ (n=1.38)\n"
    "  - Design wavelength: 555 nm (peak photopic vision)\n"
    "  - Thickness: λ/(4n) ≈ 100.5 nm"
)

# Optimal AR coating index
n_substrate = 1.5
n_optimal = np.sqrt(n_substrate)  # ≈ 1.225 (ideal), using 1.38 for MgF2
wavelength_design = 555

# Quarter-wave thickness
thickness_ar = wavelength_design / (4 * 1.38)

print(f"Optimal coating index: n = √{n_substrate} = {n_optimal:.3f}")
print("Using MgF₂: n = 1.38")
print(f"Quarter-wave thickness: {thickness_ar:.2f} nm")

# Calculate reflectance with and without AR coating
R_uncoated, T_uncoated = colour.phenomena.thin_film_tmm(
    n=[1.0, n_substrate, n_substrate],  # air | thick glass | glass substrate
    t=1000,  # Thick glass
    wavelength=wavelength_design,
    theta=0,
)

R_coated, T_coated = colour.phenomena.thin_film_tmm(
    n=[1.0, 1.38, n_substrate],  # air | MgF2 coating | glass substrate
    t=thickness_ar,
    wavelength=wavelength_design,
    theta=0,
)

print(f"\nReflectance at {wavelength_design} nm:")
print(f"  Uncoated glass: {np.mean(R_uncoated[0, 0, 0]) * 100:.2f}%")
print(f"  With AR coating: {np.mean(R_coated[0, 0, 0]) * 100:.4f}%")
reduction = (1 - np.mean(R_coated[0, 0, 0]) / np.mean(R_uncoated[0, 0, 0])) * 100
print(f"  Reduction: {reduction:.1f}%")

print("\n")

# ==============================================================================
# Multilayer Stack - Bragg Reflector
# ==============================================================================

message_box(
    "Bragg Reflector (Distributed Bragg Reflector - DBR)\n\n"
    "Multilayer quarter-wave stack:\n"
    "  - High index: n=2.0 (TiO₂)\n"
    "  - Low index: n=1.5 (SiO₂)\n"
    "  - 5 layer pairs (10 layers total)\n"
    "  - Design wavelength: 600 nm"
)

n_high = 2.0
n_low = 1.5
wl_design = 600

# Quarter-wave optical thickness for each layer
d_high = wl_design / (4 * n_high)
d_low = wl_design / (4 * n_low)

# Create 5 pairs (10 layers): air | high | low | high | low | ... | substrate
n_layers = [1.0] + [n_high, n_low] * 5 + [1.5]  # incident, 10 layers, substrate
thicknesses = [d_high, d_low] * 5  # Only layer thicknesses, not incident/substrate

print("Layer structure:")
print(f"  High-index layer: n={n_high}, d={d_high:.2f} nm")
print(f"  Low-index layer:  n={n_low}, d={d_low:.2f} nm")
print(f"  Total layers: {len(thicknesses)}")

R_bragg, T_bragg = colour.phenomena.multilayer_tmm(
    n=n_layers,
    t=thicknesses,
    wavelength=wl_design,
    theta=0,
)

print(f"\nReflectance at design wavelength ({wl_design} nm):")
print(f"  {np.mean(R_bragg[0, 0, 0]) * 100:.2f}%")
print("\nBragg mirrors achieve high reflectance through constructive")
print("interference of multiple partial reflections!")

print("\n")

# ==============================================================================
# Oblique Incidence - Polarization Effects
# ==============================================================================

message_box(
    "Oblique Incidence and Polarization\n\n"
    "Reflectance at various angles:\n"
    "  - Film: Glass (n=1.5)\n"
    "  - Thickness: 250 nm\n"
    "  - Wavelength: 550 nm"
)

angles = [0, 30, 45, 60]
print("Angle | R_s (%)  | R_p (%)  | Average (%)")
print(f"{'─' * 50}")

for angle in angles:
    R_angle, T_angle = colour.phenomena.thin_film_tmm(
        n=[1.0, 1.5, 1.0], t=250, wavelength=550, theta=angle
    )
    R_s, R_p = R_angle[0, 0, 0]  # Shape is (W, A, T, 2), extract (2,) for polarizations
    avg_R = np.mean(R_angle[0, 0, 0]) * 100
    print(f" {angle:3d}° | {R_s * 100:7.3f}  | {R_p * 100:7.3f}  | {avg_R:7.3f}")

print("\nNote: s and p polarizations differ at oblique angles!")
print("p-polarization (parallel) has lower reflectance at intermediate angles.")

print("\n")

# ==============================================================================
# Wavelength Sweep - Interference Fringes
# ==============================================================================

message_box(
    "Interference Fringes Across Wavelength\n\n"
    "Reflectance vs wavelength for a thin film:\n"
    "  - Film: n=1.5\n"
    "  - Thickness: 500 nm (thicker to show more fringes)"
)

wavelengths = SPECTRAL_SHAPE_DEFAULT.wavelengths
R_sweep, T_sweep = colour.phenomena.thin_film_tmm(
    n=[1.0, 1.5, 1.0], t=500, wavelength=wavelengths
)

# Find peaks and valleys - R_sweep shape is (W, A, T, 2)
# Take mean over polarizations
R_avg = np.mean(R_sweep[:, 0, 0, :], axis=1)  # Shape (W,)
peaks = wavelengths[
    np.where((R_avg[1:-1] > R_avg[:-2]) & (R_avg[1:-1] > R_avg[2:]))[0] + 1
]
valleys = wavelengths[
    np.where((R_avg[1:-1] < R_avg[:-2]) & (R_avg[1:-1] < R_avg[2:]))[0] + 1
]

print(f"Reflectance oscillates between {R_avg.min():.4f} and {R_avg.max():.4f}")
print("\nInterference maxima (peaks) at wavelengths:")
for peak_wl in peaks[:5]:  # Show first 5
    print(f"  ~{peak_wl:.0f} nm")

print("\nThese oscillations are characteristic of thin film interference!")

print("\n")

# ==============================================================================
# Practical Application - Optical Filter
# ==============================================================================

message_box(
    "Optical Filter Example\n\n"
    "Two-layer interference filter:\n"
    "  - Layer 1: n=1.7, d=180 nm\n"
    "  - Layer 2: n=2.3, d=120 nm\n"
    "  - Designed to reflect specific wavelengths"
)

wavelengths = SPECTRAL_SHAPE_DEFAULT.wavelengths
R_filter, T_filter = colour.phenomena.multilayer_tmm(
    n=[1.0, 1.7, 2.3, 1.5],  # air | layer1 | layer2 | glass substrate
    t=[180, 120],
    wavelength=wavelengths,
    theta=0,
)

# Find maximum reflectance wavelength - R_filter shape is (W, A, T, 2)
R_avg_filter = np.mean(
    R_filter[:, 0, 0, :], axis=1
)  # Average over polarizations, shape (W,)
max_idx = np.argmax(R_avg_filter)
max_wavelength = wavelengths[max_idx]
max_reflectance = R_avg_filter[max_idx]

print("Peak reflectance:")
print(f"  Wavelength: {max_wavelength:.0f} nm")
print(f"  Reflectance: {max_reflectance * 100:.2f}%")
print(f"\nThis filter preferentially reflects {max_wavelength:.0f} nm light!")
