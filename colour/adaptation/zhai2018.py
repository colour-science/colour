import numpy as np


def chromatic_adaptation_forward_Zhai2018(XYZb,
                                          XYZwb,
                                          Db,
                                          XYZws,
                                          Ds,
                                          XYZwo,
                                          CAT="CAT02"):

    Ywo = XYZwo[1]
    Ywb = XYZwb[1]
    Yws = XYZws[1]

    if CAT == "CAT02":
        Mt = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ])

    if CAT == "CAT16":
        Mt = np.array([
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127],
        ])

    RGBb = Mt @ XYZb
    RGBwb = Mt @ XYZwb
    RGBws = Mt @ XYZws
    RGBwo = Mt @ XYZwo

    Drgbb = Db * (Ywb / Ywo) * (RGBwo / RGBwb) + 1 - Db
    Drgbs = Ds * (Yws / Ywo) * (RGBwo / RGBws) + 1 - Ds

    Drgb = (Drgbb / Drgbs)

    RGBs = Drgb * RGBb

    XYZs = np.linalg.inv(Mt) @ RGBs

    return XYZs


"""
XYZb = np.array([48.900,43.620,6.250])
XYZwb = np.array([109.850,100,35.585])
Db = 0.9407
XYZws = np.array([95.047,100,108.883])
Ds = 0.9800
XYZwo = np.array([100,100,100])

Zhai_Luo2(XYZb, XYZwb, Db, XYZws, Ds, XYZwo, 'CAT16')
"""


def chromatic_adaptation_inverse_Zhai2018(XYZs,
                                          XYZwb,
                                          Db,
                                          XYZws,
                                          Ds,
                                          XYZwo,
                                          CAT="CAT02"):

    Ywo = XYZwo[1]
    Ywb = XYZwb[1]
    Yws = XYZws[1]

    if CAT == "CAT02":
        Mt = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ])

    if CAT == "CAT16":
        Mt = np.array([
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127],
        ])

    RGBwb = Mt @ XYZwb
    RGBws = Mt @ XYZws
    RGBwo = Mt @ XYZwo

    Drgbb = Db * (Ywb / Ywo) * (RGBwo / RGBwb) + 1 - Db
    Drgbs = Ds * (Yws / Ywo) * (RGBwo / RGBws) + 1 - Ds

    Drgb = (Drgbb / Drgbs)

    RGBs = Mt @ XYZs

    RGBb = RGBs / Drgb

    RGBs = Drgb * RGBb

    XYZb = np.linalg.inv(Mt) @ RGBb

    return XYZb


"""
XYZs = np.array([40.374,43.694,20.517])
XYZwb = np.array([109.850,100,35.585])
Db = 0.9407
XYZws = np.array([95.047,100,108.883])
Ds = 0.9800
XYZwo = np.array([100,100,100])

Zhai_Luo_inverse2(XYZs, XYZwb, Db, XYZws, Ds, XYZwo, 'CAT16')
"""
