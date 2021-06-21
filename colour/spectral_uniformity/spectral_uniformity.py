import colour
import numpy as np

def spectral_uniformity(sds):
    msds = colour.colorimetry.sds_and_msds_to_msds(sds)

    interval = msds.shape.interval

    r_i = np.gradient(np.transpose(msds.values), axis=1) / interval

    r_i = np.gradient(r_i, axis=1) / interval

    return np.mean(r_i ** 2, axis=0)
