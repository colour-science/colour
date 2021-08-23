from colour.recovery import DATA_BASIS_FUNCTIONS_DYER2017
import numpy as np
import colour


def eigen_decomposition(a, eigen_w_v_count=-1):
    A = np.dot(np.transpose(a), a)
    w, v = np.linalg.eigh(A)

    w = w[-eigen_w_v_count:]
    v = v[:, -eigen_w_v_count:]

    w = np.flipud(w)
    v = np.fliplr(v)

    return w, v


def PCA_Jiang2013(msds_camera_sensitivities,
                  eigen_w_v_count=-1,
                  additional_data=False):
    red_sensitivities, green_sensitivities, blue_sensitivities = [], [], []

    def normalised_sensitivity(msds, channel):
        return msds.signals[channel].copy().normalise().values

    for msds in msds_camera_sensitivities.values():
        red_sensitivities.append(normalised_sensitivity(msds, msds.labels[0]))
        green_sensitivities.append(
            normalised_sensitivity(msds, msds.labels[1]))
        blue_sensitivities.append(normalised_sensitivity(msds, msds.labels[2]))

    red_w_v = eigen_decomposition(
        np.vstack(red_sensitivities), eigen_w_v_count)
    green_w_v = eigen_decomposition(
        np.vstack(green_sensitivities), eigen_w_v_count)
    blue_w_v = eigen_decomposition(
        np.vstack(blue_sensitivities), eigen_w_v_count)

    if additional_data:
        return ((red_w_v[1], green_w_v[1], blue_w_v[1]),
                (red_w_v[0], green_w_v[0], blue_w_v[0]))
    else:
        return red_w_v[1], green_w_v[1], blue_w_v[1]


def recover_camera_sensitivity(RGB, illuminant, reflectances, eigen_w):
    A = []

    for reflectance in reflectances:
        A.append(np.dot(np.dot(reflectance, np.diag(illuminant)), eigen_w))

    X = np.linalg.lstsq(A, RGB)[0]
    X = np.dot(eigen_w, X)

    return X


def RGB_to_msds_camera_sensitivities_Jiang2013(
        RGB,
        illuminant,
        reflectances,
        basis_functions=DATA_BASIS_FUNCTIONS_DYER2017,
        shape=None):
    R, G, B = colour.utilities.tsplit(np.reshape(RGB, [-1, 3]))

    r_e_w, g_e_w, b_e_w = basis_functions

    if shape is None:
        shape = illuminant.shape
    elif illuminant.shape != shape:
        colour.utilities.runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}".'.format(
                illuminant.name, shape))
        # pylint: disable=E1102
        illuminant = colour.colorimetry.reshape_sd(illuminant, shape)

    if reflectances.shape != shape:
        colour.utilities.runtime_warning(
            'Aligning "{0}" reflectances shape to "{1}".'.format(
                reflectances.name, shape))
        # pylint: disable=E1102
        reflectances = colour.colorimetry.reshape_msds(reflectances, shape)

    reflectances = np.transpose(reflectances.values)

    S_R = recover_camera_sensitivity(R, illuminant.values, reflectances, r_e_w)
    S_G = recover_camera_sensitivity(G, illuminant.values, reflectances, g_e_w)
    S_B = recover_camera_sensitivity(B, illuminant.values, reflectances, b_e_w)

    msds_cam_sensitivities = colour.characterisation.RGB_CameraSensitivities(
        colour.utilities.tstack([S_R, S_G, S_B]), shape.range())

    msds_cam_sensitivities /= np.max(msds_cam_sensitivities.values)

    return msds_cam_sensitivities


# MSDS_CAMERA = RGB_to_msds_camera_sensitivities_Jiang2013(
#     RGB_REFLECTANCES, ILLUMINANT, MSDS_REFLECTANCES)
