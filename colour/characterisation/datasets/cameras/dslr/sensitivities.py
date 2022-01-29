# -*- coding: utf-8 -*-
"""
Sensitivities of *DSLR* Cameras
===============================

Defines the sensitivities of *DSLR* cameras.

Each *DSLR* camera data is in the form of a *dict* of
:class:`colour.characterisation.RGB_CameraSensitivities` classes as follows::

    {
        'name': RGB_CameraSensitivities,
        ...,
        'name': RGB_CameraSensitivities
    }

The following *DSLR* cameras are available:

-   Nikon 5100 (NPL)
-   Sigma SDMerill (NPL)

References
----------
-   :cite:`Darrodi2015a` : Darrodi, M. M., Finlayson, G., Goodman, T., &
    Mackiewicz, M. (2015). Reference data set for camera spectral sensitivity
    estimation. Journal of the Optical Society of America A, 32(3), 381.
    doi:10.1364/JOSAA.32.000381
"""

from __future__ import annotations

from functools import partial

from colour.characterisation import RGB_CameraSensitivities
from colour.hints import Dict
from colour.utilities import LazyCaseInsensitiveMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_CAMERA_SENSITIVITIES_DSLR",
    "MSDS_CAMERA_SENSITIVITIES_DSLR",
]

DATA_CAMERA_SENSITIVITIES_DSLR: Dict = {
    "Nikon 5100 (NPL)": {
        380.0: (
            0.00156384299336578000,
            0.00011500000000000000,
            0.00180956039402335990,
        ),
        385.0: (
            0.00189691771384825000,
            0.00152114360178015000,
            0.00048982814544150399,
        ),
        390.0: (
            0.00000000000000000000,
            0.00057430499183558695,
            0.00087943069176996504,
        ),
        395.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            0.00000000000000000000,
        ),
        400.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            0.00153246068848051000,
        ),
        405.0: (
            0.00071776703300973298,
            0.00119722386224553000,
            0.00569805602282062030,
        ),
        410.0: (
            0.00292397466563330000,
            0.00133571498448177000,
            0.01660828769874150200,
        ),
        415.0: (
            0.01293626801713740000,
            0.01319431696052810100,
            0.07879120559214590500,
        ),
        420.0: (
            0.04959786481566520000,
            0.06497102451249539600,
            0.36171350364994898000,
        ),
        425.0: (
            0.07607250435970400200,
            0.11510308718828900000,
            0.65970462106512295000,
        ),
        430.0: (
            0.07658892708274399300,
            0.13706582547087201000,
            0.75534360010359503000,
        ),
        435.0: (
            0.06833381956036009600,
            0.15242852584030600000,
            0.81045312707380701000,
        ),
        440.0: (
            0.06131816189646559900,
            0.16864005450745301000,
            0.87494523362472998000,
        ),
        445.0: (
            0.05473314457789760200,
            0.18329934605049600000,
            0.92671273991178704000,
        ),
        450.0: (
            0.04886204743702320100,
            0.19603263456229600000,
            0.96314088025989897000,
        ),
        455.0: (
            0.04284591974257399800,
            0.21733653278361301000,
            0.98065048133510302000,
        ),
        460.0: (
            0.04022845332691499900,
            0.25424357380995000000,
            1.00000000000000000000,
        ),
        465.0: (
            0.04340795992263239700,
            0.30864811930649899000,
            0.99640467488711104000,
        ),
        470.0: (
            0.04762021431177430200,
            0.37346871184252001000,
            0.98896988650084305000,
        ),
        475.0: (
            0.05077188480559390000,
            0.42915806139893697000,
            0.95660139953157997000,
        ),
        480.0: (
            0.05280329597225499900,
            0.45965432432137399000,
            0.90495886986980800000,
        ),
        485.0: (
            0.05257122025495090300,
            0.47106435446394301000,
            0.83940927710351598000,
        ),
        490.0: (
            0.04789463902845950100,
            0.48885616444524799000,
            0.75146259578963404000,
        ),
        495.0: (
            0.04823994170483859900,
            0.53715178104087602000,
            0.66010202032260801000,
        ),
        500.0: (
            0.05022924089718029700,
            0.61649118695883898000,
            0.56706879193613802000,
        ),
        505.0: (
            0.05507649735001429700,
            0.70700638759968903000,
            0.47935094782603899000,
        ),
        510.0: (
            0.06370211901178619900,
            0.80096424601366301000,
            0.39406273870351299000,
        ),
        515.0: (
            0.08038951305895999900,
            0.88137256686267296000,
            0.31427061879449603000,
        ),
        520.0: (
            0.10038750399831201000,
            0.93887792119838498000,
            0.24981663439426000000,
        ),
        525.0: (
            0.11861314902313400000,
            0.98446559576523596000,
            0.20182351924718100000,
        ),
        530.0: (
            0.12360875120338000000,
            1.00000000000000000000,
            0.16163395085177601000,
        ),
        535.0: (
            0.10306249932787701000,
            0.99084026557129701000,
            0.13516143147333401000,
        ),
        540.0: (
            0.07634108360672720000,
            0.96154626462922099000,
            0.10998875716043301000,
        ),
        545.0: (
            0.05278086364640900000,
            0.92814388346877297000,
            0.08639435407789379500,
        ),
        550.0: (
            0.04118873831058649700,
            0.88910231592076505000,
            0.06525313059219839400,
        ),
        555.0: (
            0.03904385351931050100,
            0.83494222924161199000,
            0.04785595345227559900,
        ),
        560.0: (
            0.04254429440089119900,
            0.77631807500187500000,
            0.03413932303860940000,
        ),
        565.0: (
            0.06021313241068020100,
            0.70731424532056497000,
            0.02401990976851929900,
        ),
        570.0: (
            0.11179621705066800000,
            0.63579620249170998000,
            0.01976793598476750100,
        ),
        575.0: (
            0.26967059703276203000,
            0.56551528450380395000,
            0.01634844781073010000,
        ),
        580.0: (
            0.56450337990639099000,
            0.49275517253522499000,
            0.01381733937020259900,
        ),
        585.0: (
            0.85360126947261405000,
            0.42475654159075799000,
            0.01195294647966710000,
        ),
        590.0: (
            0.98103242181506201000,
            0.35178931226078303000,
            0.01000909395820090100,
        ),
        595.0: (
            1.00000000000000000000,
            0.27817849879541801000,
            0.00758776308929657970,
        ),
        600.0: (
            0.96307105371259005000,
            0.21167353249961901000,
            0.00645584463521649970,
        ),
        605.0: (
            0.90552061898043101000,
            0.15671644549433000000,
            0.00522978285684488030,
        ),
        610.0: (
            0.83427841652645296000,
            0.11803962073050200000,
            0.00365998459503786990,
        ),
        615.0: (
            0.76798733762510296000,
            0.08885249534231440300,
            0.00395538505488667040,
        ),
        620.0: (
            0.70366798041157996000,
            0.07010184404853669900,
            0.00396835221654468030,
        ),
        625.0: (
            0.63916484476123703000,
            0.05690899470893220200,
            0.00349138004486036990,
        ),
        630.0: (
            0.57081292173776299000,
            0.04729879101895839700,
            0.00404302103181797010,
        ),
        635.0: (
            0.49581796193158800000,
            0.04119589002556579800,
            0.00418929985295813000,
        ),
        640.0: (
            0.43833913452368101000,
            0.03525207084991220000,
            0.00554676856500057980,
        ),
        645.0: (
            0.38896992260406899000,
            0.03069313144532450100,
            0.00546423323547744030,
        ),
        650.0: (
            0.34295621205484700000,
            0.02680396295683950100,
            0.00597382847392098970,
        ),
        655.0: (
            0.29278541836293998000,
            0.02352430119871520100,
            0.00630906774763779000,
        ),
        660.0: (
            0.23770718073119301000,
            0.02034633252474659900,
            0.00610412697742267980,
        ),
        665.0: (
            0.16491386803178501000,
            0.01545848325340879900,
            0.00483655792375416000,
        ),
        670.0: (
            0.09128771706377150600,
            0.00944075104617158980,
            0.00302664794586984980,
        ),
        675.0: (
            0.04205615047283590300,
            0.00508102204063505970,
            0.00172169700987674990,
        ),
        680.0: (
            0.02058267877678380100,
            0.00291019166901752010,
            0.00078065128657817595,
        ),
        685.0: (
            0.01028680596369610000,
            0.00162657557793382010,
            0.00056963070848184102,
        ),
        690.0: (
            0.00540759846247261970,
            0.00092251569139627796,
            0.00027523296133938200,
        ),
        695.0: (
            0.00272409261591003000,
            0.00049743349969026901,
            0.00029672137857068598,
        ),
        700.0: (
            0.00127834798711079000,
            0.00041215940263165701,
            0.00024951192304202899,
        ),
        705.0: (
            0.00078123118374132301,
            0.00031692634104666300,
            8.5000000000000006e-05,
        ),
        710.0: (
            0.00047981421940270001,
            0.00025621496960251102,
            0.00041916895092770603,
        ),
        715.0: (
            0.00049133356428571098,
            0.00000000000000000000,
            0.00015331743444139899,
        ),
        720.0: (
            0.00017414897796340199,
            0.00024353518865341200,
            1.8300000000000001e-05,
        ),
        725.0: (
            0.00012017462571764001,
            6.0200000000000000e-05,
            0.00000000000000000000,
        ),
        730.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            0.00033869381945204901,
        ),
        735.0: (
            6.1199999999999997e-05,
            0.00000000000000000000,
            0.00000000000000000000,
        ),
        740.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            0.00000000000000000000,
        ),
        745.0: (
            0.00000000000000000000,
            1.7099999999999999e-05,
            0.00016527828734010200,
        ),
        750.0: (
            0.00031099754946016501,
            5.2099999999999999e-05,
            0.00017755262214537101,
        ),
        755.0: (
            0.00000000000000000000,
            8.8499999999999996e-05,
            0.00000000000000000000,
        ),
        760.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            2.4300000000000001e-05,
        ),
        765.0: (
            0.00000000000000000000,
            0.00000000000000000000,
            6.1799999999999998e-05,
        ),
        770.0: (
            8.5599999999999994e-05,
            0.00013799999999999999,
            0.00026260703183506501,
        ),
        775.0: (
            0.00013831372865247499,
            0.0001786501727059410,
            0.00028050537004191899,
        ),
        780.0: (
            3.6199999999999999e-05,
            4.2500000000000003e-05,
            0.00000000000000000000,
        ),
    },
    "Sigma SDMerill (NPL)": {
        400.0: (
            0.00562107440608700020,
            0.00632809751263116970,
            0.16215942413307899000,
        ),
        410.0: (
            0.00650335624511722000,
            0.00976180459591275040,
            0.28549837804628603000,
        ),
        420.0: (
            0.07407911289140040000,
            0.02527177008261050100,
            0.39690431060902098000,
        ),
        430.0: (
            0.04302295946292879900,
            0.08375118585311219800,
            0.50831024317175599000,
        ),
        440.0: (
            0.03450952562247010200,
            0.14370381974360999000,
            0.62211847246948804000,
        ),
        450.0: (
            0.01889156723434350100,
            0.18361168930882199000,
            0.73742136245769496000,
        ),
        460.0: (
            0.00731107699680200000,
            0.40909478009952999000,
            0.94538036670138004000,
        ),
        470.0: (
            0.04549915123096019700,
            0.51595564086176404000,
            0.96441494770280400000,
        ),
        480.0: (
            0.05676752921111680200,
            0.60120664662705503000,
            1.00000000000000000000,
        ),
        490.0: (
            0.13419592065917799000,
            0.67031679980136305000,
            0.98598021188452500000,
        ),
        500.0: (
            0.16475268997837600000,
            0.75258747153475802000,
            0.98340266357529005000,
        ),
        510.0: (
            0.21712641978639199000,
            0.84381384368944201000,
            0.96969219567072595000,
        ),
        520.0: (
            0.30648343835824399000,
            0.90151724558812696000,
            0.94280817402079797000,
        ),
        530.0: (
            0.34984579614888500000,
            0.91975030668767699000,
            0.89664279918070899000,
        ),
        540.0: (
            0.44374258133259298000,
            0.96799429052157804000,
            0.88444590220041897000,
        ),
        550.0: (
            0.44488860528126301000,
            0.95725231064041105000,
            0.86791899071597101000,
        ),
        560.0: (
            0.47897575674702603000,
            0.95204791860047400000,
            0.83375679584908402000,
        ),
        570.0: (
            0.50950291481073895000,
            0.97628014458399803000,
            0.83204140240572999000,
        ),
        580.0: (
            0.59262909378530504000,
            0.97258624388955806000,
            0.80054956384778198000,
        ),
        590.0: (
            0.67383327560697603000,
            1.00000000000000000000,
            0.78289512474646505000,
        ),
        600.0: (
            0.71403771488106504000,
            0.96948452757777404000,
            0.73946953007191796000,
        ),
        610.0: (
            0.86000761311495100000,
            0.95441319124850699000,
            0.66718640174985699000,
        ),
        620.0: (
            0.89810302849565204000,
            0.93335435890921303000,
            0.62043627806816704000,
        ),
        630.0: (
            1.00000000000000000000,
            0.92571406833636205000,
            0.61116087876956704000,
        ),
        640.0: (
            0.99494213311245205000,
            0.88486439541503403000,
            0.55173556195710605000,
        ),
        650.0: (
            0.92085127736137995000,
            0.76165184741615699000,
            0.46538831744516401000,
        ),
        660.0: (
            0.18143311631425299000,
            0.14052437057150499000,
            0.07961907836720690000,
        ),
        670.0: (
            0.00630978795372749960,
            0.00414367215817645990,
            0.00059244446107236802,
        ),
        680.0: (
            0.00528874383171553000,
            0.00183198958165669010,
            0.00468563680483140980,
        ),
    },
}

MSDS_CAMERA_SENSITIVITIES_DSLR = LazyCaseInsensitiveMapping(
    {
        "Nikon 5100 (NPL)": partial(
            RGB_CameraSensitivities,
            DATA_CAMERA_SENSITIVITIES_DSLR["Nikon 5100 (NPL)"],
            name="Nikon 5100 (NPL)",
        ),
        "Sigma SDMerill (NPL)": partial(
            RGB_CameraSensitivities,
            DATA_CAMERA_SENSITIVITIES_DSLR["Sigma SDMerill (NPL)"],
            name="Sigma SDMerill (NPL)",
        ),
    }
)
"""
Multi-spectral distributions of *DSLR* camera sensitivities.

References
----------
:cite:`Darrodi2015a`
"""
