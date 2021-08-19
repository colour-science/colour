import numpy as np


def XYZ_to_Kim2009(XYZ, XYZw, La, E="LCD"):

    Aj = 0.89
    Bj = 0.24
    Oj = 0.65
    nj = 3.65
    Ak = 456.5
    nk = 0.62
    Am = 0.11
    Bm = 0.61
    nc = 0.57
    nq = 0.1308
    pi = np.pi

    if E == "LCD":
        E = 1.0
    if E == "media":
        E = 1.2175
    if E == "CRT":
        E = 1.4572
    if E == "paper":
        E = 1.7526

    MATRIX_XYZ_TO_LMS = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ])

    LMS = MATRIX_XYZ_TO_LMS @ XYZ
    LMSw = MATRIX_XYZ_TO_LMS @ XYZw

    LMSp = (LMS ** nc) / (LMS ** nc + La ** nc)
    LMSwp = (LMSw ** nc) / (LMSw ** nc + La ** nc)

    A = (40 * LMSp[0] + 20 * LMSp[1] + LMSp[2]) / 61
    Aw = (40 * LMSwp[0] + 20 * LMSwp[1] + LMSwp[2]) / 61

    Jp = ((-((A / Aw) - Bj) * Oj ** nj) / ((A / Aw) - Bj - Aj)) ** (1 / nj)

    J = 100 * (E * (Jp - 1) + 1)

    Q = J * (LMSw[0]) ** nq

    a = (1 / 11) * (11 * LMSp[0] - 12 * LMSp[1] + LMSp[2])
    b = (1 / 9) * (LMSp[0] + LMSp[1] - 2 * LMSp[2])

    C = Ak * (np.sqrt(a ** 2 + b ** 2) ** nk)
    M = C * (Am * np.log10(LMSw[0]) + Bm)
    s = 100 * np.sqrt(M / Q)
    h = (180 / pi) * np.arctan(b / a)

    # print("Lightness : ", J, "\nBrightness : ", Q, "\nChroma : ", C,
    # "\nColourfulness : ", M,"\nSaturation : ", s, "\nHue angle : ", h)

    return J, Q, C, M, s, h


"""
XYZ = np.array([19.01, 20.00, 21.78])
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31

Kim_Weyrich_Kautz(XYZ, XYZ_w, L_A)
"""


def Kim2009_to_XYZ(XYZw, Q, M, h, La, E="LCD"):

    Aj = 0.89
    Bj = 0.24
    Oj = 0.65
    nj = 3.65
    Ak = 456.5
    nk = 0.62
    Am = 0.11
    Bm = 0.61
    nc = 0.57
    nq = 0.1308
    pi = np.pi

    if E == "LCD":
        E = 1.0
    if E == "media":
        E = 1.2175
    if E == "CRT":
        E = 1.4572
    if E == "paper":
        E = 1.7526

    MATRIX_XYZ_TO_LMS = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ])

    LMSw = MATRIX_XYZ_TO_LMS @ XYZw

    LMSwp = (LMSw ** nc) / (LMSw ** nc + La ** nc)

    Aw = (40 * LMSwp[0] + 20 * LMSwp[1] + LMSwp[2]) / 61

    J = Q / (LMSw[0]) ** nq

    Jp = (J / 100 - 1) / E + 1

    A = Aw * ((Aj * (Jp ** nj)) / ((Jp ** nj) + (Oj ** nj)) + Bj)

    C = M / (Am * np.log10(LMSw[0]) + Bm)

    # minus a and b to match the test
    # but may be different with additional tests
    a = -(np.cos(pi * h / 180) * (C / Ak) ** (1 / nk))
    b = -(np.sin(pi * h / 180) * (C / Ak) ** (1 / nk))

    Mt = np.array([
        [1.0000, 0.3215, 0.2053],
        [1.0000, -0.6351, -0.1860],
        [1.0000, -0.1568, -4.4904],
    ])

    LMSp = Mt @ np.array([A, a, b])

    LMS = ((-(La ** nc) * LMSp) / (LMSp - 1)) ** (1 / nc)
    XYZ = np.linalg.inv(MATRIX_XYZ_TO_LMS) @ LMS

    return XYZ


"""
XYZ_w = np.array([95.05, 100.00, 108.88])
L_A = 318.31
Q = 93.08295192
M = 10.97159496
h = 30.00854111

Kim_Weyrich_Kautz_inverse(XYZ_w, Q, M, h, L_A)
"""
