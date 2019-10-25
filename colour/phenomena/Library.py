import math
import colour
import numpy as np
import matplotlib.pyplot as plt
from colour.plotting import *

# There are three different functions here: filmInten, filmColourSpec, filmColour.
# 
# filmInten takes in the thickness of a film (in nm), the wavelength of some light incident on that film (in nm), a boolean which tells it whether or not the film is made of water because if it is, it can use an empirical algorithm to determine the refractive index for
# that wavelength from the water's absolute temperature (in K) and its density (in kg/m^3). There is also the option to make "water=False" and then "n=2.4" or any other manually inserted refractive index if the liquid in question is, say, oil. The function then outputs
# a "relative intensity" of the light that will come off the film (taking into account the interference with its own partial reflections). This intensity will ALWAYS be in the range [0,1].
# 
# filmColour then takes in the same information (in the same units) but not wavelength and assumes a white-light uniform source - you may be able to add in functionality for using illuminants but I haven't read anything about how that stuff works in your library. It then
# outputs to the user's screen a plotted ColourSwatch with the colour you would see reflected from the film in that situation.
# 
# filmColourSpec takes in the same information as filmColour but instead of a single thickness, you give it a range of thicknesses in the form of thick_min and thick_max. It then plots a graph for you of how the spectrum of colours reflected varies with different
# thicknesses with a title.
# 
# Most of these things have helpful defaults built-in so you can try them straight out of the box to see what I mean! Good luck, thanks again for your help and don't be afraid to ask me for any explanation (or to completely change every part of my code)! I'm honoured
# that anyone would want it!



def filmInten(thickness,wavelength, incident=0, water=True,temperature=294, density=1000, n=1.333):
    if water == True:
        T = temperature/273.15
        r = density/1000.0
        l = wavelength/589.0
        tot = 0.244257733+0.00974634476*r-0.00373234996*T+0.000268678472*(l**2)*T+0.0015892057/(l**2)+0.00245934259/(l**2-0.229202**2)+0.90070492/(l**2-5.432937**2)-0.0166626219*(r**2)
        n = math.sqrt((2*tot+1/r)/(1/r-tot))
    return 0.5-0.5*math.cos(4*n*math.pi*thickness*math.sqrt(1-(math.sin(incident*2*math.pi/360)/n)**2)/wavelength)

def filmColourSpec(thick_min=0, thick_max=1000, incident=0, water=True, temperature=294, density=1000, n=1.333):
    raw = []
    spec = np.linspace(300,800,499)

    for t in range(thick_min,thick_max):
        raw.append(colour.sd_to_XYZ(colour.SpectralDistribution(dict([(int(i)+1,filmInten(t, i, incident, water, temperature, density, n)) for i in spec]))))

    raw = np.array([colour.XYZ_to_sRGB(i) for i in raw])
    raw /= np.max(raw)
    raw = np.clip(raw,0,1)
    figure = plt.figure()

    axes = figure.add_subplot(211)
    axes.set_xlabel('Thickness (nm)')
    axes.set_title('Spectrum of Expected Colour against Thickness')
    axes.set_yticklabels([])
    axes.set_yticks([])
    colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=c) for c in raw], height=500, axes=axes, standalone=False)
    plt.show()

def filmColour(thickness, incident=0, water=True, temperature=294, density=1000, n=1.333):
    spec = np.linspace(300,800,499)
    raw = np.array([colour.XYZ_to_sRGB(colour.sd_to_XYZ(colour.SpectralDistribution(dict([(int(i)+1,filmInten(thickness, i, incident, water, temperature, density, n)) for i in spec]))))])
    raw /= np.max(raw)
    raw = np.clip(raw,0,1)
    plot_single_colour_swatch(ColourSwatch(RGB=raw[0]))
