"""
Prague Sky Model - Wilkie et al. (2021)
=======================================

Implement the *Prague Sky Model* for physically-based sky radiance, sun
radiance, transmittance and polarisation as presented in
*Wilkie et al. (2021)* and *Vevoda et al. (2022)*.

-   :func:`colour.phenomena.sky.load_dataset_Wilkie2021`
-   :func:`colour.phenomena.sky.compute_sky_parameters_Wilkie2021`
-   :func:`colour.phenomena.sky.sky_radiance_Wilkie2021`
-   :func:`colour.phenomena.sky.sun_radiance_Wilkie2021`
-   :func:`colour.phenomena.sky.sky_polarisation_Wilkie2021`
-   :func:`colour.phenomena.sky.sky_transmittance_Wilkie2021`

References
----------
-   :cite:`Wilkie2021` : Wilkie, A., Vevoda, P., Bashford-Rogers, T., Hošek,
    L., Iser, T., Kolářová, M., Rittig, T., & Křivánek, J. (2021). A fitted
    radiance and attenuation model for realistic atmospheres. ACM Transactions
    on Graphics, 40(4), 1-14. doi:10.1145/3450626.3459758
-   :cite:`Vevoda2022` : Vévoda, P., Bashford-Rogers, T., Kolářová, M., &
    Wilkie, A. (2022). A Wide Spectral Range Sky Radiance Model. Computer
    Graphics Forum, 41(7), 291-298. doi:10.1111/cgf.14677
"""

from __future__ import annotations

import os
import struct
import typing
from dataclasses import dataclass, field

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat, NDArrayReal

from colour.algebra import lerp, linear_interpolation_index_and_factor
from colour.geometry.intersection import intersect_ray_circle_2d
from colour.utilities import (
    MixinDataclassIterable,
    Structure,
    as_float_array,
    as_int_array,
    download_url,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_COLOUR_SCIENCE",
    "RECORD_ID_ZENODO",
    "ROOT_DATASET",
    "PATH_PRAGUE_SKY_MODEL_DATASET_GROUND",
    "PATH_PRAGUE_SKY_MODEL_DATASET_FULL",
    "PATH_PRAGUE_SKY_MODEL_DATASET_SWIR",
    "CONSTANTS_WILKIE2021",
    "SUN_RAD_TABLE",
    "Metadata_SkyDataset",
    "SkyDataset_Wilkie2021",
    "SkyParameters_Wilkie2021",
    "compute_sky_parameters_Wilkie2021",
    "sky_radiance_Wilkie2021",
    "sun_radiance_Wilkie2021",
    "sky_polarisation_Wilkie2021",
    "sky_transmittance_Wilkie2021",
]

ROOT_COLOUR_SCIENCE: str = os.path.join(os.path.expanduser("~"), ".colour-science")

RECORD_ID_ZENODO: str = "19140728"
"""*Zenodo* record ID for the *Prague Sky Model* datasets."""

URL_ZENODO: str = f"https://zenodo.org/records/{RECORD_ID_ZENODO}/files"
"""*Zenodo* base URL for the *Prague Sky Model* dataset files."""

ROOT_DATASET: str = os.path.join(
    ROOT_COLOUR_SCIENCE, "colour-datasets", RECORD_ID_ZENODO, "dataset"
)

PATH_PRAGUE_SKY_MODEL_DATASET_GROUND: str = os.path.join(
    ROOT_DATASET, "PragueSkyModelDatasetGround.dat"
)
"""Path to the *Prague Sky Model* ground-level dataset."""

PATH_PRAGUE_SKY_MODEL_DATASET_FULL: str = os.path.join(
    ROOT_DATASET, "PragueSkyModelDatasetFull.dat"
)
"""Path to the *Prague Sky Model* full dataset."""

PATH_PRAGUE_SKY_MODEL_DATASET_SWIR: str = os.path.join(
    ROOT_DATASET, "PragueSkyModelDatasetSWIR.dat"
)
"""Path to the *Prague Sky Model* SWIR dataset."""

# fmt: off
SUN_RAD_TABLE: NDArrayFloat = as_float_array((
    4022.51, 4592.96, 5058.84, 5552.97, 6024.88, 6443.47, 6836.2, 7175.78,
    7380.47, 7437.44, 7391.9, 7266.06, 7121.19, 7034.24, 7024.24, 7082.31,
    7209.73, 7384.6, 7563.43, 7739.25, 7905.7, 8025.18, 8096.11, 8202.27,
    8337.15, 8602.3, 8965.52, 9316.67, 9565.64, 9814.45, 9829.41, 10184.0,
    10262.6, 10375.7, 10276.0, 10179.3, 10156.6, 10750.7, 11134.0, 11463.6,
    11860.4, 12246.2, 12524.4, 12780.0, 13187.4, 13632.4, 13985.9, 13658.3,
    13377.4, 13358.3, 13239.0, 13119.8, 13096.2, 13184.0, 13243.5, 13018.4,
    12990.4, 13159.1, 13230.8, 13258.6, 13209.9, 13343.2, 13404.8, 13305.4,
    13496.3, 13979.1, 14153.8, 14188.4, 14122.7, 13825.4, 14033.3, 13914.1,
    13837.4, 14117.2, 13982.3, 13864.5, 14118.4, 14545.7, 15029.3, 15615.3,
    15923.5, 16134.8, 16574.5, 16509.0, 16336.5, 16146.6, 15965.1, 15798.6,
    15899.8, 16125.4, 15854.3, 15986.7, 15739.7, 15319.1, 15121.5, 15220.2,
    15041.2, 14917.7, 14487.8, 14011.0, 14165.7, 14189.5, 14540.7, 14797.5,
    14641.5, 14761.6, 15153.7, 14791.8, 14907.6, 15667.4, 16313.5, 16917.0,
    17570.5, 18758.1, 20250.6, 21048.1, 21626.1, 22811.6, 23577.2, 23982.6,
    24062.1, 23917.9, 23914.1, 23923.2, 24052.6, 24228.6, 24360.8, 24629.6,
    24774.8, 24648.3, 24666.5, 24938.6, 24926.3, 24693.1, 24613.5, 24631.7,
    24569.8, 24391.5, 24245.7, 24084.4, 23713.7, 22985.4, 22766.6, 22818.9,
    22834.3, 22737.9, 22791.6, 23086.3, 23377.7, 23461.0, 23935.5, 24661.7,
    25086.9, 25520.1, 25824.3, 26198.0, 26350.2, 26375.4, 26731.2, 27250.4,
    27616.0, 28145.3, 28405.9, 28406.8, 28466.2, 28521.5, 28783.8, 29025.1,
    29082.6, 29081.3, 29043.1, 28918.9, 28871.6, 29049.0, 29152.5, 29163.2,
    29143.4, 28962.7, 28847.9, 28854.0, 28808.7, 28624.1, 28544.2, 28461.4,
    28411.1, 28478.0, 28469.8, 28513.3, 28586.5, 28628.6, 28751.5, 28948.9,
    29051.0, 29049.6, 29061.7, 28945.7, 28672.8, 28241.5, 27903.2, 27737.0,
    27590.9, 27505.6, 27270.2, 27076.2, 26929.1, 27018.2, 27206.8, 27677.2,
    27939.9, 27923.9, 27899.2, 27725.4, 27608.4, 27599.4, 27614.6, 27432.4,
    27460.4, 27392.4, 27272.0, 27299.1, 27266.8, 27386.5, 27595.9, 27586.9,
    27504.8, 27480.6, 27329.8, 26968.4, 26676.3, 26344.7, 26182.5, 26026.3,
    25900.3, 25842.9, 25885.4, 25986.5, 26034.5, 26063.5, 26216.9, 26511.4,
    26672.7, 26828.5, 26901.8, 26861.5, 26865.4, 26774.2, 26855.8, 27087.1,
    27181.3, 27183.1, 27059.8, 26834.9, 26724.3, 26759.6, 26725.9, 26724.6,
    26634.5, 26618.5, 26560.1, 26518.7, 26595.3, 26703.2, 26712.7, 26733.9,
    26744.3, 26764.4, 26753.2, 26692.7, 26682.7, 26588.1, 26478.0, 26433.7,
    26380.7, 26372.9, 26343.3, 26274.7, 26162.3, 26160.5, 26210.0, 26251.2,
    26297.9, 26228.9, 26222.3, 26269.7, 26295.6, 26317.9, 26357.5, 26376.1,
    26342.4, 26303.5, 26276.7, 26349.2, 26390.0, 26371.6, 26346.7, 26327.6,
    26274.2, 26247.3, 26228.7, 26152.1, 25910.3, 25833.2, 25746.5, 25654.3,
    25562.0, 25458.8, 25438.0, 25399.1, 25324.3, 25350.0, 25514.0, 25464.9,
    25398.5, 25295.2, 25270.2, 25268.4, 25240.6, 25184.9, 25149.6, 25123.9,
    25080.3, 25027.9, 25012.3, 24977.9, 24852.6, 24756.4, 24663.5, 24483.6,
    24398.6, 24362.6, 24325.1, 24341.7, 24288.7, 24284.2, 24257.3, 24178.8,
    24097.6, 24175.6, 24175.7, 24139.7, 24088.1, 23983.2, 23902.7, 23822.4,
    23796.2, 23796.9, 23814.5, 23765.5, 23703.0, 23642.0, 23592.6, 23552.0,
    23514.6, 23473.5, 23431.0, 23389.3, 23340.0, 23275.1, 23187.3, 23069.5,
    22967.0, 22925.3, 22908.9, 22882.5, 22825.0, 22715.4, 22535.5, 22267.1,
    22029.4, 21941.6, 21919.5, 21878.8, 21825.6, 21766.0, 21728.9, 21743.2,
    21827.1, 21998.7, 22159.4, 22210.0, 22187.2, 22127.2, 22056.2, 22000.2,
    21945.9, 21880.2, 21817.1, 21770.3, 21724.3, 21663.2, 21603.3, 21560.4,
    21519.8, 21466.2, 21401.6, 21327.7, 21254.2, 21190.7, 21133.6, 21079.3,
    21024.0, 20963.7, 20905.5, 20856.6, 20816.6, 20785.2, 20746.7, 20685.3,
    20617.8, 20561.1, 20500.4, 20421.2, 20333.4, 20247.0, 20175.3, 20131.4,
    20103.2, 20078.5, 20046.8, 19997.2, 19952.9, 19937.2, 19930.8, 19914.4,
    19880.8, 19823.0, 19753.8, 19685.9, 19615.3, 19537.5, 19456.8, 19377.6,
    19309.4, 19261.9, 19228.0, 19200.5, 19179.5, 19164.8, 19153.1, 19140.6,
    19129.2, 19120.6, 19104.5, 19070.6, 19023.9, 18969.3, 18911.4, 18855.0,
    18798.6, 18740.8, 18672.7, 18585.2, 18501.0, 18442.4, 18397.5, 18353.9,
    18313.2, 18276.8, 18248.3, 18231.2, 18224.0, 18225.4, 18220.1, 18192.6,
    18155.1, 18119.8, 18081.6, 18035.6, 17987.4, 17942.8, 17901.7, 17864.2,
    17831.1, 17802.9, 17771.5, 17728.6, 17669.7, 17590.1, 17509.5, 17447.4,
    17396.0, 17347.4, 17300.3, 17253.2, 17206.1, 17159.0, 17127.6, 17127.6,
    17133.6, 17120.4, 17097.2, 17073.3, 17043.7, 17003.4, 16966.3, 16946.3,
    16930.9, 16907.7, 16882.7, 16862.0, 16837.8, 16802.1, 16759.2, 16713.6,
    16661.8, 16600.8, 16542.6, 16499.4, 16458.7, 16408.0, 16360.6, 16329.5,
    16307.4, 16286.7, 16264.9, 16239.6, 16207.8, 16166.8, 16118.2, 16064.0,
    16011.2, 15966.9, 15931.9, 15906.9, 15889.1, 15875.5, 15861.2, 15841.3,
    15813.1, 15774.2, 15728.8, 15681.4, 15630.0, 15572.9, 15516.5, 15467.2,
    15423.0, 15381.6, 15354.4, 15353.0, 15357.3, 15347.3, 15320.2, 15273.1,
    15222.0, 15183.1, 15149.6, 15114.6, 15076.8, 15034.6, 14992.9, 14996.7,
    14959.9, 14915.2, 14861.9, 14810.6, 14762.1, 14708.4, 14657.6, 14594.7,
    14508.8, 14423.0, 14344.9, 14271.9, 14219.1, 14189.1, 14159.1, 14128.6,
    14100.4, 14071.0, 14037.9, 13995.2, 13943.4, 13891.4, 13842.4, 13804.1,
    13782.4, 13791.1, 13821.6, 13853.0, 13877.6, 13898.1, 13900.4, 13880.8,
    13859.8, 13837.4, 13810.9, 13783.0, 13757.6, 13751.4, 13764.6, 13780.8,
    13797.3, 13811.4, 13806.5, 13783.0, 13757.3, 13727.3, 13695.7, 13662.4,
    13626.9, 13593.5, 13564.5, 13535.3, 13506.1, 13477.7, 13445.7, 13407.6,
    13367.9, 13327.3, 13284.9, 13240.2, 13195.4, 13146.7, 13093.4, 13038.5,
    12982.6, 12926.5, 12873.9, 12823.0, 12770.6, 12720.0, 12670.5, 12623.1,
    12579.7, 12536.1, 12489.0, 12438.8, 12384.7, 12327.6, 12271.9, 12222.9,
    12180.0, 12142.7, 12110.1, 12079.9, 12047.1, 12014.1, 11981.5, 11948.4,
    11916.6, 11883.9, 11847.8, 11813.5, 11782.2, 11754.1, 11731.6, 11711.5,
    11688.3, 11664.0, 11638.5, 11609.9, 11581.5, 11551.8, 11519.0, 11482.2,
    11443.1, 11403.0, 11363.2, 11324.4, 11287.6, 11254.0, 11222.4, 11193.7,
    11167.0, 11144.3, 11124.3, 11104.7, 11084.5, 11064.4, 11042.3, 11019.8,
    10999.1, 10983.3, 10971.7, 10963.4, 10957.1, 10954.6, 10953.0, 10947.9,
    10940.3, 10933.5, 10925.5, 10917.1, 10912.7, 10911.2, 10909.5, 10908.5,
    10907.8, 10909.0, 10913.3, 10915.2, 10913.6, 10912.4, 10909.5, 10904.4,
    10902.5, 10904.6, 10906.8, 10906.5, 10904.3, 10898.1, 10887.1, 10872.4,
    10856.5, 10839.2, 10822.5, 10804.5, 10784.0, 10763.0, 10741.9, 10716.9,
    10689.9, 10662.9, 10633.9, 10603.0, 10575.1, 10551.6, 10530.6, 10512.0,
    10495.8, 10480.2, 10463.3, 10445.1, 10425.5, 10404.6, 10383.5, 10363.6,
    10344.7, 10326.9, 10310.2, 10293.6, 10276.4, 10258.5, 10239.8, 10220.5,
    10200.9, 10181.4, 10162.1, 10142.8, 10123.7, 10104.2, 10083.6, 10062.0,
    10039.4, 10015.8, 9991.44, 9966.56, 9941.17, 9915.27, 9888.87, 9862.37,
    9836.19, 9810.32, 9784.77, 9759.55, 9735.05, 9711.69, 9689.47, 9668.4,
    9648.47, 9628.83, 9608.61, 9587.82, 9566.46, 9544.54, 9522.7, 9501.63,
    9481.32, 9461.77, 9442.98, 9424.13, 9404.39, 9383.76, 9362.24, 9339.84,
    9317.15, 9294.77, 9272.72, 9250.98, 9229.56, 9208.1, 9186.27, 9164.05,
    9141.46, 9118.48, 9095.66, 9073.54, 9052.12, 9031.4, 9011.37, 8992.33,
    8974.56, 8958.06, 8942.82, 8928.86, 8915.56, 8902.33, 8889.16, 8876.05,
    8863.01, 8850.03, 8837.11, 8824.26, 8811.47, 8798.74, 8785.76, 8772.21,
    8758.09, 8743.4, 8728.13, 8712.52, 8696.78, 8680.91, 8664.91, 8648.79,
    8632.61, 8616.42, 8600.24, 8584.05, 8567.87, 8551.78, 8535.88, 8520.17,
    8504.65, 8489.32, 8474.06, 8458.73, 8443.34, 8427.88, 8412.36, 8397.03,
    8382.15, 8367.71, 8353.71, 8340.16, 8326.52, 8312.23, 8297.32, 8281.77,
    8265.58, 8248.95, 8232.07, 8214.93, 8197.54, 8179.9, 8162.35, 8145.24,
    8128.58, 8112.36, 8096.59, 8081.04, 8065.49, 8049.94, 8034.39, 8018.84,
    8003.29, 7987.74, 7972.19, 7956.64, 7941.08, 7925.95, 7911.63, 7898.15,
    7885.48, 7873.65, 7862.06, 7850.16, 7837.94, 7825.41, 7812.56, 7799.29,
    7785.52, 7771.24, 7756.45, 7741.15, 7725.73, 7710.56, 7695.64, 7680.98,
    7666.57, 7652.29, 7638.01, 7623.73, 7609.45, 7595.17, 7580.95, 7566.86,
    7552.89, 7539.06, 7525.35, 7511.22, 7496.15, 7480.12, 7463.14, 7445.21,
    7427.19, 7409.92, 7393.42, 7377.68, 7362.7, 7348.26, 7334.14, 7320.33,
    7306.85, 7293.68, 7280.92, 7268.67, 7256.93, 7245.69, 7234.96, 7224.14,
    7212.62, 7200.4, 7187.49, 7173.87, 7160.26, 7147.34, 7135.12, 7123.6,
    7112.78, 7102.47, 7092.47, 7082.79, 7073.43, 7064.38, 7055.31, 7045.85,
    7036.01, 7025.79, 7015.19, 7004.69, 6994.76, 6985.39, 6976.6, 6968.38,
    6960.29, 6951.88, 6943.15, 6934.11, 6924.75, 6915.54, 6906.97, 6899.04,
    6891.74, 6885.08, 6878.54, 6871.62, 6864.32, 6856.64, 6848.58, 6839.38,
    6828.27, 6815.26, 6800.34, 6783.52, 6766.16, 6749.63, 6733.92, 6719.04,
    6704.98, 6691.36, 6677.81, 6664.32, 6650.9, 6637.54, 6624.56, 6612.28,
    6600.69, 6589.81, 6579.62, 6569.43, 6558.55, 6546.96, 6534.68, 6521.7,
    6508.44, 6495.3, 6482.29, 6469.4, 6456.64, 6444.46, 6433.29, 6423.13,
    6413.99, 6405.87, 6397.87, 6389.11, 6379.59, 6369.31, 6358.26, 6347.19,
    6336.81, 6327.13, 6318.15, 6309.87, 6302.25, 6295.27, 6288.92, 6283.21,
    6278.13, 6273.24, 6268.1, 6262.71, 6257.06, 6251.15, 6245.0, 6238.59,
    6231.92, 6225.0, 6217.83, 6210.09, 6201.46, 6191.94, 6181.53, 6170.23,
    6158.9, 6148.39, 6138.71, 6129.86, 6121.83, 6113.96, 6105.58, 6096.7,
    6087.3, 6077.4, 6066.93, 6055.82, 6044.08, 6031.7, 6018.69, 6005.42,
    5992.29, 5979.27, 5966.39, 5953.63, 5940.81, 5927.74, 5914.41, 5900.82,
    5886.99, 5873.09, 5859.31, 5845.67, 5832.15, 5818.75, 5805.58, 5792.73,
    5780.2, 5767.98, 5756.08, 5744.3, 5732.47, 5720.56, 5708.6, 5696.57,
    5684.26, 5671.44, 5658.11, 5644.27, 5629.93, 5615.46, 5601.24, 5587.27,
    5573.56, 5560.11, 5546.94, 5534.08, 5521.55, 5509.33, 5497.43, 5485.72,
    5474.07, 5462.49, 5450.97, 5439.51, 5428.15, 5416.92, 5405.81, 5394.83,
    5383.97, 5373.12, 5362.14, 5351.03, 5339.8, 5328.44, 5317.2, 5306.35,
    5295.88, 5285.78, 5276.07, 5266.58, 5257.16, 5247.8, 5238.5, 5229.26,
    5220.09, 5210.98, 5201.94, 5192.96, 5184.04, 5175.28, 5166.78, 5158.52,
    5150.53, 5142.78, 5135.13, 5127.42, 5119.65, 5111.81, 5103.91, 5095.97,
    5088.04, 5080.1, 5072.17, 5064.24, 5056.59, 5049.51, 5043.01, 5037.07,
    5031.71, 5026.28, 5020.16, 5013.33, 5005.81, 4997.59, 4989.02, 4980.45,
    4971.89, 4963.32, 4954.75, 4946.08, 4937.23, 4928.19, 4918.95, 4909.52,
    4899.85, 4889.85, 4879.53, 4868.9, 4857.95, 4847.1, 4836.75, 4826.92,
    4817.59, 4808.76, 4799.81, 4790.1, 4779.63, 4768.4, 4756.4, 4744.4,
    4733.17, 4722.7, 4712.99, 4704.04, 4695.31, 4686.26, 4676.9, 4667.22,
    4657.23, 4646.91, 4636.28, 4625.33, 4614.06, 4602.48, 4590.96, 4579.89,
    4569.25, 4559.07, 4549.32, 4539.87, 4530.54, 4521.33, 4512.26, 4503.31,
    4494.39, 4485.41, 4476.36, 4467.26, 4458.08, 4449.32, 4441.45, 4434.47,
    4428.38, 4423.17, 4418.16, 4412.64, 4406.61, 4400.07, 4393.03, 4386.11,
    4379.95, 4374.55, 4369.92, 4366.05, 4362.21, 4357.67, 4352.43, 4346.5,
    4339.87, 4332.92, 4326.03, 4319.21, 4312.45, 4305.75, 4299.09, 4292.42,
    4285.76, 4279.09, 4272.43, 4265.76, 4259.1, 4252.44, 4245.77, 4239.11,
    4232.7, 4226.79, 4221.4, 4216.51, 4212.13, 4207.91, 4203.5, 4198.9,
    4194.11, 4189.12, 4183.63, 4177.32, 4170.18, 4162.21, 4153.42, 4144.34,
    4135.52, 4126.95, 4118.64, 4110.58, 4102.74, 4095.09, 4087.63, 4080.36,
    4073.29, 4066.31, 4059.32, 4052.34, 4045.36, 4038.38, 4031.52, 4024.92,
    4018.57, 4012.48, 4006.64, 4000.83, 3994.84, 3988.65, 3982.27, 3975.7,
    3968.97, 3962.12, 3955.13, 3948.03, 3940.79, 3933.27, 3925.3, 3916.89,
    3908.04, 3898.74, 3889.44, 3880.59, 3872.18, 3864.21, 3856.69, 3849.52,
    3842.6, 3835.94, 3829.52, 3823.37, 3817.21, 3810.8, 3804.14, 3797.22,
    3790.05, 3782.59, 3774.81, 3766.72, 3758.31, 3749.58, 3740.73, 3731.94,
    3723.21, 3714.55, 3705.95, 3697.44, 3689.06, 3680.81, 3672.69, 3664.69,
    3656.69, 3648.57, 3640.32, 3631.94, 3623.43, 3615.12, 3607.31, 3600.01,
    3593.22, 3586.94, 3580.97, 3575.13, 3569.42, 3563.83, 3558.37, 3552.92,
    3547.33, 3541.62, 3535.78, 3529.81, 3524.04, 3518.77, 3514.01, 3509.75,
    3506.01, 3502.52, 3499.03, 3495.54, 3492.05, 3488.56, 3485.13, 3481.83,
    3478.65, 3475.61, 3472.69, 3469.74, 3466.59, 3463.26, 3459.74, 3456.03,
    3452.22, 3448.41, 3444.6, 3440.79, 3436.98, 3433.3, 3429.88, 3426.7,
    3423.78, 3421.12, 3418.51, 3415.79, 3412.93, 3409.95, 3406.84, 3403.66,
    3400.49, 3397.32, 3394.14, 3390.97, 3387.76, 3384.49, 3381.16, 3377.77,
    3374.31, 3370.66, 3366.69, 3362.41, 3357.8, 3352.88, 3347.65, 3342.09,
    3336.22, 3330.04, 3323.53, 3317.02, 3310.84, 3304.96, 3299.41, 3294.17,
    3289.13, 3284.15, 3279.23, 3274.37, 3269.58, 3264.79, 3259.93, 3255.01,
    3250.03, 3244.98, 3239.78, 3234.32, 3228.61, 3222.64, 3216.42, 3210.17,
    3204.11, 3198.24, 3192.56, 3187.07, 3181.54, 3175.77, 3169.74, 3163.45,
    3156.92, 3150.09, 3142.95, 3135.49, 3127.72, 3119.63, 3111.69, 3104.39,
    3097.73, 3091.7, 3086.3, 3081.26, 3076.28, 3071.36, 3066.5, 3061.71,
    3056.63, 3050.92, 3044.57, 3037.59, 3029.97, 3021.91, 3013.6, 3005.03,
    2996.21, 2987.13, 2977.89, 2968.6, 2959.23, 2949.81, 2940.32, 2931.02,
    2922.17, 2913.76, 2905.79, 2898.27, 2890.69, 2882.53, 2873.8, 2864.5,
    2854.63, 2844.7, 2835.21, 2826.17, 2817.57, 2809.41, 2801.82, 2794.94,
    2788.75, 2783.26, 2778.47, 2773.58, 2767.8, 2761.14, 2753.59, 2745.14,
    2736.42, 2728.01, 2719.91, 2712.14, 2704.68, 2697.51, 2690.59, 2683.93,
    2677.52, 2671.36, 2665.39, 2659.55, 2653.84, 2648.26, 2642.8, 2637.37,
    2631.88, 2626.33, 2620.71, 2615.03, 2609.22, 2603.22, 2597.03, 2590.66,
    2584.09, 2577.58, 2571.39, 2565.52, 2559.97, 2554.73, 2549.72, 2544.83,
    2540.07, 2535.43, 2530.93, 2526.29, 2521.28, 2515.89, 2510.11, 2503.95,
    2497.73, 2491.77, 2486.05, 2480.6, 2475.39, 2470.12, 2464.47, 2458.44,
    2452.03, 2445.24, 2438.42, 2431.91, 2425.72, 2419.85, 2414.3, 2409.0,
    2403.89, 2398.97, 2394.24, 2389.7, 2385.23, 2380.69, 2376.09, 2371.42,
    2366.7, 2361.97, 2357.3, 2352.7, 2348.16, 2343.69, 2339.09, 2334.17,
    2328.93, 2323.38, 2317.51, 2311.44, 2305.32, 2299.13, 2292.88, 2286.56,
    2280.28, 2274.12, 2268.09, 2262.19, 2256.41, 2250.64, 2244.74, 2238.71,
    2232.55, 2226.26, 2220.08, 2214.21, 2208.65, 2203.42, 2198.5, 2193.67,
    2188.72, 2183.64, 2178.44, 2173.11, 2167.74, 2162.44, 2157.21, 2152.03,
    2146.93, 2141.69, 2136.14, 2130.26, 2124.08, 2117.57, 2110.91, 2104.24,
    2097.58, 2090.91, 2084.25, 2077.93, 2072.31, 2067.4, 2063.17, 2059.65,
    2056.32, 2052.67, 2048.7, 2044.42, 2039.82, 2035.09, 2030.42, 2025.82,
    2021.28, 2016.81, 2012.49, 2008.43, 2004.62, 2001.07, 1997.77, 1994.53,
    1991.17, 1987.68, 1984.06, 1980.31, 1976.66, 1973.33, 1970.32, 1967.62,
    1965.24, 1963.08, 1961.05, 1959.14, 1957.37, 1955.72, 1954.32, 1953.31,
    1952.67, 1952.42, 1952.54, 1952.8, 1952.92, 1952.92, 1952.8, 1952.54,
    1952.0, 1951.02, 1949.59, 1947.72, 1945.4, 1942.93, 1940.58, 1938.36,
    1936.26, 1934.3, 1932.39, 1930.49, 1928.58, 1926.68, 1924.77, 1922.78,
    1920.59, 1918.21, 1915.64, 1912.87, 1910.11, 1907.54, 1905.16, 1902.97,
    1900.97, 1898.94, 1896.66, 1894.12, 1891.33, 1888.28, 1884.91, 1881.17,
    1877.04, 1872.54, 1867.65, 1862.76, 1858.26, 1854.13, 1850.39, 1847.02,
    1843.88, 1840.8, 1837.79, 1834.84, 1831.95, 1829.16, 1826.49, 1823.95,
    1821.54, 1819.25, 1817.16, 1815.32, 1813.73, 1812.4, 1811.32, 1810.27,
    1809.03, 1807.61, 1805.99, 1804.18, 1802.27, 1800.37, 1798.47, 1796.56,
    1794.66, 1792.75, 1790.85, 1788.95, 1787.04, 1785.14, 1783.33, 1781.71,
    1780.28, 1779.04, 1778.0, 1776.95, 1775.71, 1774.28, 1772.67, 1770.86,
    1768.76, 1766.29, 1763.43, 1760.19, 1756.58, 1752.77, 1748.96, 1745.15,
    1741.34, 1737.53, 1733.63, 1729.54, 1725.25, 1720.78, 1716.11, 1711.13,
    1705.7, 1699.83, 1693.52, 1686.76, 1680.14, 1674.25, 1669.1, 1664.67,
    1660.97, 1657.64, 1654.31, 1650.97, 1647.64, 1644.31, 1640.95, 1637.52,
    1634.03, 1630.47, 1626.86, 1623.21, 1619.56, 1615.91, 1612.26, 1608.61,
    1605.01, 1601.5, 1598.09, 1594.77, 1591.55, 1588.38, 1585.2, 1582.03,
    1578.85, 1575.68, 1572.43, 1569.02, 1565.45, 1561.72, 1557.83, 1553.86,
    1549.9, 1545.93, 1541.96, 1538.0, 1534.06, 1530.19, 1526.38, 1522.64,
    1518.95, 1515.3, 1511.65, 1508.0, 1504.36, 1500.71, 1496.99, 1493.15,
    1489.19, 1485.09, 1480.87, 1476.59, 1472.3, 1468.02, 1463.73, 1459.45,
    1455.3, 1451.41, 1447.79, 1444.43, 1441.33, 1438.37, 1435.41, 1432.45,
    1429.49, 1426.52, 1423.56, 1420.6, 1417.64, 1414.68, 1411.71, 1408.74,
    1405.75, 1402.73, 1399.7, 1396.64, 1393.57, 1390.5, 1387.44, 1384.37,
    1381.3, 1378.23, 1375.16, 1372.1, 1369.03, 1365.96, 1362.91, 1359.91,
    1356.95, 1354.03, 1351.15, 1348.29, 1345.44, 1342.58, 1339.73, 1336.87,
    1334.01, 1331.16, 1328.3, 1325.44, 1322.59, 1319.71, 1316.79, 1313.83,
    1310.83, 1307.78, 1304.71, 1301.64, 1298.58, 1295.51, 1292.44, 1289.37,
    1286.3, 1283.24, 1280.17, 1277.1, 1274.05, 1271.05, 1268.09, 1265.17,
    1262.29, 1259.43, 1256.58, 1253.72, 1250.87, 1248.01, 1245.15, 1242.3,
    1239.44, 1236.58, 1233.73, 1230.88, 1228.06, 1225.25, 1222.47, 1219.71,
    1216.96, 1214.21, 1211.46, 1208.71, 1205.96, 1203.21, 1200.46, 1197.71,
    1194.96, 1192.21, 1189.49, 1186.83, 1184.24, 1181.71, 1179.25, 1176.82,
    1174.38, 1171.95, 1169.52, 1167.08, 1164.65, 1162.22, 1159.78, 1157.35,
    1154.92, 1152.51, 1150.14, 1147.81, 1145.52, 1143.28, 1141.06, 1138.84,
    1136.62, 1134.4, 1132.17, 1129.95, 1127.73, 1125.51, 1123.29, 1121.07,
    1118.9, 1116.83, 1114.88, 1113.03, 1111.28, 1109.59, 1107.9, 1106.2,
    1104.51, 1102.82, 1101.13, 1099.43, 1097.74, 1096.05, 1094.36, 1092.66,
    1090.97, 1089.28, 1087.59, 1085.89, 1084.2, 1082.51, 1080.81, 1079.12,
    1077.43, 1075.74, 1074.04, 1072.35, 1070.66, 1068.97, 1067.28, 1065.62,
    1063.98, 1062.37, 1060.77, 1059.18, 1057.59, 1056.01, 1054.42, 1052.83,
    1051.25, 1049.66, 1048.07, 1046.49, 1044.9, 1043.26, 1041.5, 1039.64,
    1037.66, 1035.58, 1033.44, 1031.29, 1029.15, 1027.01, 1024.87, 1022.73,
    1020.58, 1018.44, 1016.3, 1014.16, 1012.01, 1009.87, 1007.73, 1005.59,
    1003.45, 1001.34, 999.3, 997.33, 995.43, 993.59, 991.8, 990.0,
    988.2, 986.4, 984.6, 982.8, 981.01, 979.21, 977.41, 975.61,
    973.93, 972.48, 971.26, 970.28, 969.53, 968.89, 968.26, 967.62,
    966.99, 966.35, 965.72, 965.09, 964.45, 963.82, 963.18, 962.55,
    961.91, 961.28, 960.64, 960.01, 959.28, 958.36, 957.25, 955.95,
    954.45, 952.87, 951.28, 949.69, 948.11, 946.52, 944.93, 943.35,
    941.76, 940.17, 938.59, 937.0, 935.41, 933.83, 932.24, 930.65,
    929.06, 927.45, 925.82, 924.18, 922.52, 920.85, 919.19, 917.52,
    915.85, 914.19, 912.52, 910.86, 909.19, 907.52, 905.86, 904.19,
    902.53, 900.86, 899.19, 897.53, 895.91, 894.39, 892.96, 891.62,
    890.39, 889.2, 888.01, 886.82, 885.63, 884.44, 883.25, 882.06,
    880.87, 879.68, 878.49, 877.3, 876.11, 874.92, 873.73, 872.54,
    871.31, 870.0, 868.61, 867.14, 865.59, 864.01, 862.42, 860.83,
    859.25, 857.66, 856.07, 854.49, 852.9, 851.31, 849.73, 848.14,
    846.55, 844.97, 843.38, 841.79, 840.24, 838.74, 837.32, 835.95,
    834.65, 833.38, 832.11, 830.84, 829.57, 828.3, 827.03, 825.76,
    824.5, 823.23, 821.96, 820.69, 819.42, 818.15, 816.88, 815.61,
    814.37, 813.2, 812.09, 811.04, 810.06, 809.1, 808.15, 807.2,
    806.25, 805.3, 804.34, 803.39, 802.44, 801.49, 800.53, 799.58,
    798.63, 797.68, 796.73, 795.77, 794.87, 794.06, 793.35, 792.73,
    792.2, 791.73, 791.25, 790.78, 790.3, 789.82, 789.35, 788.87,
    788.4, 787.92, 787.44, 786.97, 786.49, 786.02, 785.54, 785.06,
    784.61, 784.19, 783.81, 783.48, 783.18, 782.9, 782.62, 782.35,
    782.07, 781.79, 781.51, 781.24, 780.96, 780.68, 780.4, 780.12,
    779.85, 779.57, 779.29, 779.01, 778.68, 778.23, 777.66, 776.98,
    776.18, 775.32, 774.46, 773.61, 772.75, 771.89, 771.04, 770.18,
    769.32, 768.47, 767.61, 766.75, 765.9, 765.04, 764.18, 763.32,
    762.47, 761.61, 760.75, 759.9, 759.04, 758.25, 757.59, 757.07,
    756.68, 756.42, 756.23, 756.04, 755.85, 755.66, 755.47, 755.28,
    755.09, 754.9, 754.71, 754.52, 754.33, 754.14, 753.95, 753.76,
    753.57, 753.38, 753.19, 752.99, 752.8, 752.61, 752.43, 752.25,
    752.08, 751.91, 751.76, 751.6, 751.45, 751.3, 751.15, 751.0,
    750.84, 750.69, 750.54, 750.39, 750.2,
))
# fmt: on

CONSTANTS_WILKIE2021: Structure = Structure(
    planet_radius=6378000.0,
    safety_altitude=50.0,
    sun_radius=0.004654793,
    atmosphere_width=100000.0,
    distance_to_edge=1571524.413613,
    sun_radiance_start=280.0,
    sun_radiance_step=1.0,
    sun_radiance_end=280.0 + 1.0 * len(SUN_RAD_TABLE),
    low_altitude_threshold=0.3,
    distance_epsilon=1e-10,
)
"""
Constants for the *Prague Sky Model*.

-   ``planet_radius``: Earth radius in meters.
-   ``safety_altitude``: Safety altitude shift in meters.
-   ``sun_radius``: Angular radius of the sun in radians (0.2667 degrees).
-   ``atmosphere_width``: Width of the atmosphere in meters.
-   ``distance_to_edge``: Maximum distance to the edge of the atmosphere
    in the transmittance model.
-   ``sun_radiance_start``: Start wavelength of the solar radiance table
    in nanometers.
-   ``sun_radiance_step``: Wavelength step of the solar radiance table
    in nanometers.
-   ``sun_radiance_end``: End wavelength of the solar radiance table
    in nanometers.
-   ``low_altitude_threshold``: Altitude threshold in meters below which
    the transmittance model uses a simplified ray-atmosphere intersection
    to avoid numerical issues.
-   ``distance_epsilon``: Minimum distance used to guard against division
    by zero when computing the transmittance distance parameter.
"""


@dataclass
class Metadata_SkyDataset:
    """
    Internal metadata for radiance/polarisation coefficient layout.

    Parameters
    ----------
    rank
        Rank of the tensor decomposition.
    sun_offset
        Offset to the sun coefficients within a single rank block.
    sun_stride
        Stride between consecutive rank blocks (sun + zenith breaks).
    sun_break_points
        Break points for piecewise linear approximation of the sun
        (gamma) dimension.
    zenith_offset
        Offset to the zenith coefficients within a single rank block.
    zenith_stride
        Stride between consecutive rank blocks for the zenith dimension.
    zenith_break_points
        Break points for piecewise linear approximation of the zenith
        (alpha/zero) dimension.
    emphasis_offset
        Offset to the emphasis coefficients (radiance only).
    emphasis_break_points
        Break points for the emphasis de-weighting function (radiance
        only, empty for polarisation).
    total_coefficients_single_configuration
        Total number of coefficients for a single configuration
        (one channel, one visibility/albedo/altitude/elevation).
    total_coefficients_all_configurations
        Total number of coefficients across all configurations.
    """

    rank: int = 0
    sun_offset: int = 0
    sun_stride: int = 0
    sun_break_points: NDArrayFloat = field(default_factory=lambda: zeros(0))
    zenith_offset: int = 0
    zenith_stride: int = 0
    zenith_break_points: NDArrayFloat = field(default_factory=lambda: zeros(0))
    emphasis_offset: int = 0
    emphasis_break_points: NDArrayFloat = field(default_factory=lambda: zeros(0))
    total_coefficients_single_configuration: int = 0
    total_coefficients_all_configurations: int = 0


@dataclass
class SkyDataset_Wilkie2021(MixinDataclassIterable):
    """
    Implement support for loading and holding a *Prague Sky Model* dataset.

    Parameters
    ----------
    path
        Path to the ``.dat`` dataset file.

    Attributes
    ----------
    -   :attr:`~SkyDataset_Wilkie2021.path`
    -   :attr:`~SkyDataset_Wilkie2021.channels`
    -   :attr:`~SkyDataset_Wilkie2021.channel_start`
    -   :attr:`~SkyDataset_Wilkie2021.channel_width`

    Methods
    -------
    -   :meth:`~SkyDataset_Wilkie2021.__init__`
    -   :meth:`~SkyDataset_Wilkie2021.read`

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`

    Examples
    --------
    >>> dataset = SkyDataset_Wilkie2021()  # doctest: +SKIP
    >>> dataset.channels  # doctest: +SKIP
    11
    """

    path: str = PATH_PRAGUE_SKY_MODEL_DATASET_GROUND
    channels: int = field(default=0, init=False)
    channel_start: float = field(default=0.0, init=False)
    channel_width: float = field(default=0.0, init=False)
    visibilities_radiance: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    albedos_radiance: NDArrayFloat = field(default_factory=lambda: zeros(0), init=False)
    altitudes_radiance: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    elevations_radiance: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    metadata_radiance: Metadata_SkyDataset = field(
        default_factory=Metadata_SkyDataset, init=False
    )
    data_radiance: NDArrayFloat = field(default_factory=lambda: zeros(0), init=False)
    metadata_polarisation: Metadata_SkyDataset = field(
        default_factory=Metadata_SkyDataset, init=False
    )
    data_polarisation: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    altitude_dimension: int = field(default=0, init=False)
    distance_dimension: int = field(default=0, init=False)
    rank_transmittance: int = field(default=0, init=False)
    altitudes_transmittance: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    visibilities_transmittance: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    data_transmittance_u: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )
    data_transmittance_v: NDArrayFloat = field(
        default_factory=lambda: zeros(0), init=False
    )

    def __post_init__(self) -> None:
        """
        Read the dataset from :attr:`path` if provided.

        If the file does not exist and the path matches one of the
        default dataset paths, the dataset is automatically downloaded
        from *Zenodo*.
        """

        if self.path:
            if not os.path.isfile(self.path) and self.path.startswith(ROOT_DATASET):
                download_url(
                    f"{URL_ZENODO}/{os.path.basename(self.path)}?download=1",
                    filename=self.path,
                )
            self.read()

    def read(self) -> SkyDataset_Wilkie2021:
        """
        Read the *Prague Sky Model* dataset from the file at
        :attr:`path`.

        Returns
        -------
        :class:`SkyDataset_Wilkie2021`
            *self* for chaining.
        """

        with open(self.path, "rb") as f:
            # Radiance section.
            self.visibilities_radiance = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()
            self.albedos_radiance = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()
            self.altitudes_radiance = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()
            self.elevations_radiance = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()

            self.channels = struct.unpack("<i", f.read(4))[0]
            self.channel_start = struct.unpack("<d", f.read(8))[0]
            self.channel_width = struct.unpack("<d", f.read(8))[0]

            configurations = (
                self.channels
                * len(self.elevations_radiance)
                * len(self.altitudes_radiance)
                * len(self.albedos_radiance)
                * len(self.visibilities_radiance)
            )

            metadata_radiance = Metadata_SkyDataset()
            metadata_radiance.rank = struct.unpack("<i", f.read(4))[0]

            metadata_radiance.sun_break_points = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()
            metadata_radiance.zenith_break_points = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()
            metadata_radiance.emphasis_break_points = np.frombuffer(
                f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
            ).copy()

            metadata_radiance.sun_offset = 0
            metadata_radiance.sun_stride = len(
                metadata_radiance.sun_break_points
            ) + len(metadata_radiance.zenith_break_points)
            metadata_radiance.zenith_offset = len(metadata_radiance.sun_break_points)
            metadata_radiance.zenith_stride = metadata_radiance.sun_stride
            metadata_radiance.emphasis_offset = (
                metadata_radiance.rank * metadata_radiance.sun_stride
            )
            metadata_radiance.total_coefficients_single_configuration = (
                metadata_radiance.emphasis_offset
                + len(metadata_radiance.emphasis_break_points)
            )
            metadata_radiance.total_coefficients_all_configurations = (
                metadata_radiance.total_coefficients_single_configuration
                * configurations
            )

            self.metadata_radiance = metadata_radiance

            data_radiance = np.zeros(
                metadata_radiance.total_coefficients_all_configurations,
                dtype=np.float32,
            )
            offset = 0

            for _configuration in range(configurations):
                for _rank_index in range(metadata_radiance.rank):
                    sun_coefficients = as_float_array(
                        np.frombuffer(
                            f.read(len(metadata_radiance.sun_break_points) * 2),
                            dtype="<f2",
                        )
                    )
                    zenith_scale = struct.unpack("<d", f.read(8))[0]
                    zenith_coefficients = as_float_array(
                        np.frombuffer(
                            f.read(len(metadata_radiance.zenith_break_points) * 2),
                            dtype="<f2",
                        )
                    )

                    data_radiance[
                        offset : offset + len(metadata_radiance.sun_break_points)
                    ] = as_float_array(sun_coefficients, dtype=np.float32)
                    offset += len(metadata_radiance.sun_break_points)

                    scaled = as_float_array(
                        zenith_coefficients / zenith_scale, dtype=np.float32
                    )
                    data_radiance[
                        offset : offset + len(metadata_radiance.zenith_break_points)
                    ] = scaled
                    offset += len(metadata_radiance.zenith_break_points)

                emphasis_coefficients = as_float_array(
                    np.frombuffer(
                        f.read(len(metadata_radiance.emphasis_break_points) * 2),
                        dtype="<f2",
                    )
                )
                data_radiance[
                    offset : offset + len(metadata_radiance.emphasis_break_points)
                ] = as_float_array(emphasis_coefficients, dtype=np.float32)
                offset += len(metadata_radiance.emphasis_break_points)

            self.data_radiance = data_radiance

            # Transmittance section.
            self.distance_dimension = struct.unpack("<i", f.read(4))[0]
            self.altitude_dimension = struct.unpack("<i", f.read(4))[0]

            visibility_count = struct.unpack("<i", f.read(4))[0]
            altitude_count = struct.unpack("<i", f.read(4))[0]
            self.rank_transmittance = struct.unpack("<i", f.read(4))[0]

            self.altitudes_transmittance = as_float_array(
                np.frombuffer(f.read(altitude_count * 4), dtype="<f4").copy()
            )
            self.visibilities_transmittance = as_float_array(
                np.frombuffer(f.read(visibility_count * 4), dtype="<f4").copy()
            )

            total_u = (
                self.distance_dimension
                * self.altitude_dimension
                * self.rank_transmittance
                * altitude_count
            )
            total_v = (
                visibility_count
                * self.rank_transmittance
                * self.channels
                * altitude_count
            )

            self.data_transmittance_u = np.frombuffer(
                f.read(total_u * 4), dtype="<f4"
            ).copy()
            self.data_transmittance_v = np.frombuffer(
                f.read(total_v * 4), dtype="<f4"
            ).copy()

            # Polarisation section (optional).
            rank_bytes = f.read(4)
            if len(rank_bytes) >= 4:
                rank = struct.unpack("<i", rank_bytes)[0]

                metadata_polarisation = Metadata_SkyDataset()
                metadata_polarisation.rank = rank

                metadata_polarisation.sun_break_points = np.frombuffer(
                    f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
                ).copy()
                metadata_polarisation.zenith_break_points = np.frombuffer(
                    f.read(struct.unpack("<i", f.read(4))[0] * 8), dtype="<f8"
                ).copy()

                metadata_polarisation.emphasis_break_points = zeros(0)
                metadata_polarisation.sun_offset = 0
                metadata_polarisation.sun_stride = len(
                    metadata_polarisation.sun_break_points
                ) + len(metadata_polarisation.zenith_break_points)
                metadata_polarisation.zenith_offset = len(
                    metadata_polarisation.sun_break_points
                )
                metadata_polarisation.zenith_stride = metadata_polarisation.sun_stride
                metadata_polarisation.emphasis_offset = 0
                metadata_polarisation.total_coefficients_single_configuration = (
                    metadata_polarisation.rank * metadata_polarisation.sun_stride
                )
                metadata_polarisation.total_coefficients_all_configurations = (
                    metadata_polarisation.total_coefficients_single_configuration
                    * configurations
                )

                coefficient_count = (
                    len(metadata_polarisation.sun_break_points)
                    + len(metadata_polarisation.zenith_break_points)
                ) * metadata_polarisation.rank
                data_polarisation = np.zeros(
                    metadata_polarisation.total_coefficients_all_configurations,
                    dtype=np.float32,
                )
                offset = 0

                for _configuration in range(configurations):
                    chunk = np.frombuffer(
                        f.read(coefficient_count * 4), dtype="<f4"
                    ).copy()
                    data_polarisation[offset : offset + coefficient_count] = chunk
                    offset += coefficient_count

                self.metadata_polarisation = metadata_polarisation
                self.data_polarisation = data_polarisation

        return self


@dataclass
class SkyParameters_Wilkie2021(MixinDataclassIterable):
    """
    Hold computed parameters for querying the *Prague Sky Model*.

    Parameters
    ----------
    theta
        Angle between view direction and zenith in radians [0, pi].
    gamma
        Angle between view direction and sun in radians [0, pi].
    shadow
        Altitude-corrected shadow angle in radians [0, pi].
    zero
        Altitude-corrected zenith angle in radians [0, pi].
    elevation
        Sun elevation at view point in radians.
    altitude
        Observer altitude in meters.
    visibility
        Meteorological range at ground level in kilometers.
    albedo
        Ground albedo [0, 1].

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    theta: float | NDArrayFloat = 0.0
    gamma: float | NDArrayFloat = 0.0
    shadow: float | NDArrayFloat = 0.0
    zero: float | NDArrayFloat = 0.0
    elevation: float | NDArrayFloat = 0.0
    altitude: float | NDArrayFloat = 0.0
    visibility: float = 0.0
    albedo: float = 0.0


def compute_sky_parameters_Wilkie2021(
    view_point: ArrayLike,
    view_direction: ArrayLike,
    sun_elevation: float,
    sun_azimuth: float,
    visibility: float,
    albedo: float,
) -> SkyParameters_Wilkie2021:
    """
    Compute parameters for querying the *Prague Sky Model*.

    Supports batched *view_direction* inputs.  If *view_direction* has
    shape ``(..., 3)``, the returned parameter fields will have shape
    ``(...)``.

    Parameters
    ----------
    view_point
        Observer position ``[x, y, z]`` in meters with z up.
    view_direction
        View direction(s) ``[..., 3]``, unit-length normalised.
    sun_elevation
        Sun elevation at ground level at origin in radians.
    sun_azimuth
        Sun azimuth at ground level at origin in radians.
    visibility
        Meteorological range at ground level in kilometers.
    albedo
        Ground albedo [0, 1].

    Returns
    -------
    :class:`SkyParameters_Wilkie2021`
        Computed parameters.  Fields are scalars when *view_direction*
        is ``(3,)`` or arrays matching the batch dimensions otherwise.

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    view_point_array = as_float_array(view_point)
    view_direction_array = as_float_array(view_direction)
    view_direction_array = view_direction_array / np.linalg.norm(
        view_direction_array, axis=-1, keepdims=True
    )

    center = np.array([0.0, 0.0, -CONSTANTS_WILKIE2021.planet_radius])
    to_view_point = view_point_array - center
    to_view_point_normalised = to_view_point / np.linalg.norm(to_view_point)
    distance_to_view = (
        np.linalg.norm(to_view_point) + CONSTANTS_WILKIE2021.safety_altitude
    )
    to_shifted_view_point = to_view_point_normalised * distance_to_view
    shifted_view_point = center + to_shifted_view_point

    altitude = max(distance_to_view - CONSTANTS_WILKIE2021.planet_radius, 0.0)

    # Direction to sun.
    direction_to_sun = np.array(
        [
            np.cos(sun_azimuth) * np.cos(sun_elevation),
            np.sin(sun_azimuth) * np.cos(sun_elevation),
            np.sin(sun_elevation),
        ]
    )

    # Solar elevation at view point.
    dot_zenith_sun = float(np.dot(to_view_point_normalised, direction_to_sun))
    elevation = 0.5 * np.pi - np.arccos(dot_zenith_sun)

    # Altitude-corrected view direction.
    if distance_to_view > CONSTANTS_WILKIE2021.planet_radius:
        look_at_point = shifted_view_point + view_direction_array
        correction = (
            np.sqrt(distance_to_view**2 - CONSTANTS_WILKIE2021.planet_radius**2)
            / distance_to_view
        )
        to_new_origin = to_view_point_normalised * (distance_to_view - correction)
        new_origin = center + to_new_origin
        correct_view = look_at_point - new_origin
        correct_view_n = correct_view / np.linalg.norm(
            correct_view, axis=-1, keepdims=True
        )
    else:
        correct_view_n = view_direction_array

    # Gamma (sun angle) - no correction.  Shape: (...,).
    gamma = np.arccos(
        np.clip(np.sum(view_direction_array * direction_to_sun, axis=-1), -1, 1)
    )

    # Shadow angle - requires correction.
    shadow_angle = sun_elevation + np.pi * 0.5
    shadow_direction = np.array(
        [
            np.cos(shadow_angle) * np.cos(sun_azimuth),
            np.cos(shadow_angle) * np.sin(sun_azimuth),
            np.sin(shadow_angle),
        ]
    )
    shadow = np.arccos(
        np.clip(np.sum(correct_view_n * shadow_direction, axis=-1), -1, 1)
    )

    # Zenith angle (corrected and uncorrected).  Shape: (...,).
    zero = np.arccos(
        np.clip(np.sum(correct_view_n * to_view_point_normalised, axis=-1), -1, 1)
    )
    theta = np.arccos(
        np.clip(np.sum(view_direction_array * to_view_point_normalised, axis=-1), -1, 1)
    )

    return SkyParameters_Wilkie2021(
        theta=theta,
        gamma=gamma,
        shadow=shadow,
        zero=zero,
        elevation=float(elevation),
        altitude=float(altitude),
        visibility=visibility,
        albedo=albedo,
    )


def _reconstruct_sky_model(
    data: NDArrayFloat,
    offsets: NDArrayReal,
    gamma_index: NDArrayReal,
    gamma_factor: NDArrayReal,
    alpha_index: NDArrayReal,
    alpha_factor: NDArrayReal,
    zero_index: NDArrayReal,
    zero_factor: NDArrayReal,
    metadata: Metadata_SkyDataset,
) -> NDArrayFloat:
    """
    Reconstruct radiance/polarisation via inverse tensor decomposition.

    Parameters
    ----------
    offsets
        Coefficient offsets, shape ``(*S, W)``.
    gamma_index, gamma_factor
        Sun angle interpolation parameters, shape ``(*S,)``.
    alpha_index, alpha_factor
        Zenith/shadow angle interpolation parameters, shape ``(*S,)``.
    zero_index, zero_factor
        Emphasis angle interpolation parameters, shape ``(*S,)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Reconstructed values, shape ``(*S, W)``.
    """

    result = np.zeros(offsets.shape, dtype=np.float64)

    for rank_index in range(metadata.rank):
        sun_index = as_int_array(
            offsets
            + metadata.sun_offset
            + rank_index * metadata.sun_stride
            + gamma_index[..., None]
        )
        sun_value = lerp(
            gamma_factor[..., None],
            as_float_array(data[sun_index]),
            as_float_array(data[sun_index + 1]),
        )

        zenith_index = as_int_array(
            offsets
            + metadata.zenith_offset
            + rank_index * metadata.zenith_stride
            + alpha_index[..., None]
        )
        zenith_value = lerp(
            alpha_factor[..., None],
            as_float_array(data[zenith_index]),
            as_float_array(data[zenith_index + 1]),
        )

        result += sun_value * zenith_value

    if len(metadata.emphasis_break_points) > 0:
        emphasis_index = as_int_array(
            offsets + metadata.emphasis_offset + zero_index[..., None]
        )
        emphasis_value = lerp(
            zero_factor[..., None],
            as_float_array(data[emphasis_index]),
            as_float_array(data[emphasis_index + 1]),
        )
        result *= emphasis_value
        # The emphasis term is non-negative by construction in the
        # *Prague Sky Model* reference; clipping floors out fitting noise.
        np.maximum(result, 0.0, out=result)

    return result


def _evaluate_sky_model(
    dataset: SkyDataset_Wilkie2021,
    parameters: SkyParameters_Wilkie2021,
    wavelength: NDArrayFloat,
    data: NDArrayFloat,
    metadata: Metadata_SkyDataset,
) -> NDArrayFloat:
    """
    Evaluate the model for given wavelength.

    If the parameter fields have shape ``(*S,)`` and *wavelength* has
    shape ``(W,)``, the result has shape ``(*S, W)``.
    """

    wavelength = np.atleast_1d(wavelength)
    wavelength_count = len(wavelength)

    # Angle parameters.
    gamma_index, gamma_factor = linear_interpolation_index_and_factor(
        parameters.gamma, metadata.sun_break_points
    )

    elevation = as_float_array(parameters.elevation)
    shadow = as_float_array(parameters.shadow)
    zero = as_float_array(parameters.zero)

    if len(metadata.emphasis_break_points) > 0:
        alpha_value = np.where(elevation < 0, shadow, zero)
        alpha_index, alpha_factor = linear_interpolation_index_and_factor(
            alpha_value, metadata.zenith_break_points
        )
        zero_index, zero_factor = linear_interpolation_index_and_factor(
            zero, metadata.emphasis_break_points
        )
    else:
        alpha_index, alpha_factor = linear_interpolation_index_and_factor(
            zero, metadata.zenith_break_points
        )
        zero_index = np.zeros_like(gamma_index)
        zero_factor = np.zeros_like(gamma_factor)

    # Configuration parameters.
    visibility_index, visibility_factor = linear_interpolation_index_and_factor(
        parameters.visibility, dataset.visibilities_radiance
    )
    albedo_index, albedo_factor = linear_interpolation_index_and_factor(
        parameters.albedo, dataset.albedos_radiance
    )
    altitude_index, altitude_factor = linear_interpolation_index_and_factor(
        parameters.altitude, dataset.altitudes_radiance
    )
    elevation_index, elevation_factor = linear_interpolation_index_and_factor(
        np.degrees(elevation), dataset.elevations_radiance
    )

    # Broadcast all parameter arrays to the common batch shape.
    shape = np.broadcast_shapes(
        np.shape(gamma_index),
        np.shape(alpha_index),
        np.shape(zero_index),
        np.shape(visibility_index),
        np.shape(albedo_index),
        np.shape(altitude_index),
        np.shape(elevation_index),
    )
    gamma_index = np.broadcast_to(gamma_index, shape)
    gamma_factor = np.broadcast_to(gamma_factor, shape)
    alpha_index = np.broadcast_to(alpha_index, shape)
    alpha_factor = np.broadcast_to(alpha_factor, shape)
    zero_index = np.broadcast_to(zero_index, shape)
    zero_factor = np.broadcast_to(zero_factor, shape)
    visibility_index = np.broadcast_to(visibility_index, shape)
    visibility_factor = np.broadcast_to(visibility_factor, shape)
    albedo_index = np.broadcast_to(albedo_index, shape)
    albedo_factor = np.broadcast_to(albedo_factor, shape)
    altitude_index = np.broadcast_to(altitude_index, shape)
    altitude_factor = np.broadcast_to(altitude_factor, shape)
    elevation_index = np.broadcast_to(elevation_index, shape)
    elevation_factor = np.broadcast_to(elevation_factor, shape)

    # Filter wavelength within dataset range.
    wavelength_end = dataset.channel_start + dataset.channels * dataset.channel_width
    valid = (wavelength >= dataset.channel_start) & (wavelength < wavelength_end)
    if not np.any(valid):
        return np.zeros((*shape, wavelength_count), dtype=np.float64)

    channel_indices = as_int_array(
        np.floor((wavelength[valid] - dataset.channel_start) / dataset.channel_width)
    )
    n_valid = len(channel_indices)

    # Precompute strides for offset calculation.
    elevation_count = len(dataset.elevations_radiance)
    altitude_count = len(dataset.altitudes_radiance)
    albedo_count = len(dataset.albedos_radiance)
    total_coefficients_single = metadata.total_coefficients_single_configuration

    # 4D hypercube corners over (visibility, albedo, altitude, elevation).
    grid_shape = (16, *shape, n_valid)
    grid = np.zeros(grid_shape, dtype=np.float64)

    for i in range(16):
        visibility_grid_index = np.minimum(
            visibility_index + i // 8, len(dataset.visibilities_radiance) - 1
        )
        albedo_grid_index = np.minimum(albedo_index + (i % 8) // 4, albedo_count - 1)
        altitude_grid_index = np.minimum(
            altitude_index + (i % 4) // 2, altitude_count - 1
        )
        elevation_grid_index = np.minimum(elevation_index + i % 2, elevation_count - 1)

        offsets = total_coefficients_single * (
            channel_indices
            + dataset.channels * elevation_grid_index[..., None]
            + dataset.channels * elevation_count * altitude_grid_index[..., None]
            + dataset.channels
            * elevation_count
            * altitude_count
            * albedo_grid_index[..., None]
            + dataset.channels
            * elevation_count
            * altitude_count
            * albedo_count
            * visibility_grid_index[..., None]
        )

        grid[i] = _reconstruct_sky_model(
            data,
            offsets,
            gamma_index,
            gamma_factor,
            alpha_index,
            alpha_factor,
            zero_index,
            zero_factor,
            metadata,
        )

    # 4-level hierarchical interpolation (elevation, altitude, albedo,
    # visibility).
    factors = [elevation_factor, altitude_factor, albedo_factor, visibility_factor]
    for level_factor in factors:
        low = grid[0::2]
        high = grid[1::2]
        grid = low + level_factor[..., None] * (high - low)

    result_valid = grid[0]

    # Place valid wavelength into full result.
    result = np.zeros((*shape, wavelength_count), dtype=np.float64)
    result[..., valid] = result_valid

    return result


def sky_radiance_Wilkie2021(
    dataset: SkyDataset_Wilkie2021,
    parameters: SkyParameters_Wilkie2021,
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Compute sky radiance (without direct sun) for given wavelength.

    Parameters
    ----------
    dataset
        Loaded dataset from :func:`load_dataset_Wilkie2021`.
    parameters
        Sky parameters from :func:`compute_sky_parameters_Wilkie2021`.
    wavelength
        Wavelengths in nanometers.

    Returns
    -------
    :class:`numpy.ndarray`
        Sky radiance values.

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    return _evaluate_sky_model(
        dataset,
        parameters,
        as_float_array(wavelength),
        dataset.data_radiance,
        dataset.metadata_radiance,
    )


def sun_radiance_Wilkie2021(
    dataset: SkyDataset_Wilkie2021,
    parameters: SkyParameters_Wilkie2021,
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Compute sun radiance (without inscattered sky contribution) for given
    wavelength.

    Parameters
    ----------
    dataset
        Loaded dataset from :func:`load_dataset_Wilkie2021`.
    parameters
        Sky parameters from :func:`compute_sky_parameters_Wilkie2021`.
    wavelength
        Wavelengths in nanometers.

    Returns
    -------
    :class:`numpy.ndarray`
        Sun radiance values (zero for directions not hitting the sun).

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    wavelength = as_float_array(wavelength)
    gamma = as_float_array(parameters.gamma)
    shape = np.shape(gamma)

    wavelength_count = len(wavelength)
    result = np.zeros((*shape, wavelength_count), dtype=np.float64)

    # Mask: only directions hitting the sun disk.
    hits_sun = gamma <= CONSTANTS_WILKIE2021.sun_radius

    if not np.any(hits_sun):
        return result

    valid_wavelength = (wavelength >= CONSTANTS_WILKIE2021.sun_radiance_start) & (
        wavelength < CONSTANTS_WILKIE2021.sun_radiance_end
    )
    if not np.any(valid_wavelength):
        return result

    # Interpolate solar radiance from table.
    index_float = (
        wavelength[valid_wavelength] - CONSTANTS_WILKIE2021.sun_radiance_start
    ) / CONSTANTS_WILKIE2021.sun_radiance_step
    index_integer = as_int_array(np.floor(index_float))
    index_fraction = index_float - index_integer
    index_integer = np.clip(index_integer, 0, len(SUN_RAD_TABLE) - 2)

    sun_radiance_value = (
        SUN_RAD_TABLE[index_integer] * (1.0 - index_fraction)
        + SUN_RAD_TABLE[index_integer + 1] * index_fraction
    )

    # Compute transmittance towards the sun.
    transmittance = sky_transmittance_Wilkie2021(
        dataset, parameters, wavelength[valid_wavelength], np.inf
    )

    result[..., valid_wavelength] = sun_radiance_value * transmittance
    result *= hits_sun[..., None]

    return result


def sky_polarisation_Wilkie2021(
    dataset: SkyDataset_Wilkie2021,
    parameters: SkyParameters_Wilkie2021,
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Compute degree of polarisation for given wavelength.

    Parameters
    ----------
    dataset
        Loaded dataset from :func:`load_dataset_Wilkie2021`.
    parameters
        Sky parameters from :func:`compute_sky_parameters_Wilkie2021`.
    wavelength
        Wavelengths in nanometers.

    Returns
    -------
    :class:`numpy.ndarray`
        Degree of polarisation. The result is negated from the internal
        model coefficients following the convention of the reference C++
        implementation.

    Raises
    ------
    ValueError
        If the dataset does not contain polarisation data.

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    if dataset.metadata_polarisation.rank == 0:
        message = "The supplied dataset does not contain polarisation data."
        raise ValueError(message)

    return -_evaluate_sky_model(
        dataset,
        parameters,
        as_float_array(wavelength),
        dataset.data_polarisation,
        dataset.metadata_polarisation,
    )


def _compute_transmittance_interpolation(
    value: NDArrayFloat, count: int, power: int
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Compute transmittance-specific interpolation index and factor."""

    value = as_float_array(value)
    index = np.minimum(as_int_array(value * count), count - 1)
    lower = index / count
    upper = (index + 1) / count
    denominator = upper**power - lower**power
    factor = np.where(
        (index < count - 1) & (denominator != 0),
        np.clip(
            (value**power - lower**power)
            / np.where(denominator == 0, 1.0, denominator),
            0.0,
            1.0,
        ),
        0.0,
    )
    return index, factor


def _compute_transmittance_parameters(
    dataset: SkyDataset_Wilkie2021,
    theta: ArrayLike,
    distance: float,
    altitude: ArrayLike,
) -> tuple[tuple[NDArrayFloat, NDArrayFloat], tuple[NDArrayFloat, NDArrayFloat]]:
    """Convert viewing geometry to transmittance interpolation parameters."""

    theta = as_float_array(theta)
    altitude = as_float_array(altitude)

    ray_direction_x = np.sin(theta)
    ray_direction_y = np.cos(theta)
    ray_position_y = CONSTANTS_WILKIE2021.planet_radius + altitude

    shape = np.broadcast_shapes(np.shape(theta), np.shape(altitude))
    ray_origin = np.stack(
        [np.broadcast_to(0.0, shape), np.broadcast_to(ray_position_y, shape)],
        axis=-1,
    )
    ray_direction = np.stack(
        [
            np.broadcast_to(ray_direction_x, shape),
            np.broadcast_to(ray_direction_y, shape),
        ],
        axis=-1,
    )

    atmosphere_edge = (
        CONSTANTS_WILKIE2021.planet_radius + CONSTANTS_WILKIE2021.atmosphere_width
    )

    is_low_altitude = altitude < CONSTANTS_WILKIE2021.low_altitude_threshold

    # Low altitude: atmosphere edge if theta <= pi/2, else 0.
    distance_low_atmosphere = intersect_ray_circle_2d(
        ray_origin, ray_direction, atmosphere_edge
    )
    distance_low = np.where(theta <= 0.5 * np.pi, distance_low_atmosphere, 0.0)

    # High altitude: planet first, atmosphere edge if planet missed.
    distance_planet = intersect_ray_circle_2d(
        ray_origin, ray_direction, CONSTANTS_WILKIE2021.planet_radius
    )
    distance_atmosphere = intersect_ray_circle_2d(
        ray_origin, ray_direction, atmosphere_edge
    )
    distance_high = np.where(
        np.isnan(distance_planet), distance_atmosphere, distance_planet
    )

    distance_to_intersection = np.where(is_low_altitude, distance_low, distance_high)
    distance_to_intersection = np.minimum(distance_to_intersection, distance)

    intersection_x = ray_direction_x * distance_to_intersection
    intersection_y = ray_direction_y * distance_to_intersection + ray_position_y

    intersection_distance = np.sqrt(
        intersection_x * intersection_x + intersection_y * intersection_y
    )

    altitude_parameter = np.clip(
        intersection_distance - CONSTANTS_WILKIE2021.planet_radius,
        0.0,
        CONSTANTS_WILKIE2021.atmosphere_width,
    )
    altitude_parameter = (
        altitude_parameter / CONSTANTS_WILKIE2021.atmosphere_width
    ) ** (1.0 / 3.0)

    distance_parameter = (
        np.arccos(
            np.clip(
                intersection_y
                / np.maximum(
                    intersection_distance, CONSTANTS_WILKIE2021.distance_epsilon
                ),
                -1,
                1,
            )
        )
        * CONSTANTS_WILKIE2021.planet_radius
    )
    distance_parameter = np.sqrt(
        distance_parameter / CONSTANTS_WILKIE2021.distance_to_edge
    )
    distance_parameter = np.sqrt(distance_parameter)
    distance_parameter = np.minimum(1.0, distance_parameter)

    altitude_interpolation = _compute_transmittance_interpolation(
        altitude_parameter, dataset.altitude_dimension, 3
    )
    distance_interpolation = _compute_transmittance_interpolation(
        distance_parameter, dataset.distance_dimension, 4
    )

    return altitude_interpolation, distance_interpolation


def _reconstruct_transmittance(
    dataset: SkyDataset_Wilkie2021,
    visibility_index: NDArrayReal,
    altitude_index: NDArrayReal,
    altitude_interpolation: tuple[NDArrayReal, NDArrayReal],
    distance_interpolation: tuple[NDArrayReal, NDArrayReal],
    channel_indices: NDArrayReal,
) -> NDArrayFloat:
    """
    Reconstruct transmittance for multiple channels.

    If *visibility_index* and *altitude_index* have shape ``(*S,)`` and
    *channel_indices* has shape ``(W,)``, the result has shape
    ``(*S, W)``.
    """

    altitude_i, altitude_f = altitude_interpolation
    distance_i, distance_f = distance_interpolation

    altitude_d = dataset.altitude_dimension
    distance_d = dataset.distance_dimension
    rank = dataset.rank_transmittance
    altitude_transmittance_count = len(dataset.altitudes_transmittance)

    # V-coefficients offset: (*S, W, rank)
    v_coefficient_base = (
        (
            visibility_index[..., None] * altitude_transmittance_count
            + altitude_index[..., None]
        )
        * dataset.channels
        + channel_indices
    ) * rank

    v_coefficient_offsets = v_coefficient_base[..., None] + np.arange(rank)
    v_coefficients = as_float_array(
        dataset.data_transmittance_v[as_int_array(v_coefficient_offsets)]
    )

    # U-coefficients: iterate 2x2 grid (altitude x distance).
    transmittance = np.zeros(
        (*np.shape(altitude_i), len(channel_indices), 4), dtype=np.float64
    )

    grid_index = 0
    for altitude_offset in range(2):
        a = altitude_i + altitude_offset
        altitude_valid = a < altitude_d
        for distance_offset in range(2):
            d = distance_i + distance_offset
            distance_valid = d < distance_d

            u_coefficient_base = (
                altitude_index[..., None] * altitude_d * distance_d * rank
                + (d[..., None] * altitude_d + a[..., None]) * rank
            )
            u_coefficient_offsets = u_coefficient_base[..., None] + np.arange(rank)

            safe_u_offsets = np.clip(
                u_coefficient_offsets, 0, len(dataset.data_transmittance_u) - 1
            )
            u_coefficients = as_float_array(
                dataset.data_transmittance_u[as_int_array(safe_u_offsets)]
            )

            dot = np.sum(u_coefficients * v_coefficients, axis=-1)
            mask = altitude_valid & distance_valid
            transmittance[..., grid_index] = np.where(mask[..., None], dot, 0.0)
            grid_index += 1

    # Bilinear interpolation over distance then altitude.
    low = lerp(distance_f[..., None], transmittance[..., 0], transmittance[..., 1])
    high = lerp(distance_f[..., None], transmittance[..., 2], transmittance[..., 3])
    low = np.maximum(low, 0.0)
    high = np.maximum(high, 0.0)

    return lerp(altitude_f[..., None], low, high)


def sky_transmittance_Wilkie2021(
    dataset: SkyDataset_Wilkie2021,
    parameters: SkyParameters_Wilkie2021,
    wavelength: ArrayLike,
    distance: float = np.inf,
) -> NDArrayFloat:
    """
    Compute atmospheric transmittance for given wavelength and distance.

    Parameters
    ----------
    dataset
        Loaded dataset.
    parameters
        Sky parameters from :func:`compute_sky_parameters_Wilkie2021`.
    wavelength
        Wavelengths in nanometers.
    distance
        Distance along view direction in meters (``np.inf`` for infinity).

    Returns
    -------
    :class:`numpy.ndarray`
        Transmittance values [0, 1].

    References
    ----------
    :cite:`Wilkie2021`, :cite:`Vevoda2022`
    """

    wavelength = as_float_array(wavelength)
    wavelength_count = len(wavelength)

    theta = as_float_array(parameters.theta)
    shape = np.shape(theta)

    wavelength_end = dataset.channel_start + dataset.channels * dataset.channel_width
    valid = (wavelength >= dataset.channel_start) & (wavelength < wavelength_end)
    if not np.any(valid):
        return np.zeros((*shape, wavelength_count), dtype=np.float64)

    channel_indices = as_int_array(
        np.floor((wavelength[valid] - dataset.channel_start) / dataset.channel_width)
    )

    visibility_index, visibility_factor = linear_interpolation_index_and_factor(
        parameters.visibility, dataset.visibilities_transmittance
    )
    altitude_index, altitude_factor = linear_interpolation_index_and_factor(
        parameters.altitude, dataset.altitudes_transmittance
    )

    # Broadcast config indices to common shape.
    shape = np.broadcast_shapes(
        np.shape(theta),
        np.shape(visibility_index),
        np.shape(altitude_index),
    )
    visibility_index = np.broadcast_to(visibility_index, shape)
    visibility_factor = np.broadcast_to(visibility_factor, shape)
    altitude_index = np.broadcast_to(altitude_index, shape)
    altitude_factor = np.broadcast_to(altitude_factor, shape)

    altitude_interpolation, distance_interpolation = _compute_transmittance_parameters(
        dataset, parameters.theta, distance, parameters.altitude
    )

    # 4-point interpolation over visibility x altitude.
    transmittance = _reconstruct_transmittance(
        dataset,
        visibility_index,
        altitude_index,
        altitude_interpolation,
        distance_interpolation,
        channel_indices,
    )

    transmittance_altitude_high = _reconstruct_transmittance(
        dataset,
        visibility_index,
        np.minimum(altitude_index + 1, len(dataset.altitudes_transmittance) - 1),
        altitude_interpolation,
        distance_interpolation,
        channel_indices,
    )
    transmittance = lerp(
        altitude_factor[..., None], transmittance, transmittance_altitude_high
    )

    transmittance_visibility_high = _reconstruct_transmittance(
        dataset,
        np.minimum(visibility_index + 1, len(dataset.visibilities_transmittance) - 1),
        altitude_index,
        altitude_interpolation,
        distance_interpolation,
        channel_indices,
    )
    transmittance_visibility_altitude_high = _reconstruct_transmittance(
        dataset,
        np.minimum(visibility_index + 1, len(dataset.visibilities_transmittance) - 1),
        np.minimum(altitude_index + 1, len(dataset.altitudes_transmittance) - 1),
        altitude_interpolation,
        distance_interpolation,
        channel_indices,
    )
    transmittance_visibility_high = lerp(
        altitude_factor[..., None],
        transmittance_visibility_high,
        transmittance_visibility_altitude_high,
    )
    transmittance = lerp(
        visibility_factor[..., None], transmittance, transmittance_visibility_high
    )

    # Transmittance is stored as square root.
    transmittance = transmittance * transmittance
    transmittance = np.clip(transmittance, 0.0, 1.0)

    result = np.zeros((*shape, wavelength_count), dtype=np.float64)
    result[..., valid] = transmittance

    return result
