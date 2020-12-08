/*=========================================================================

Program: FEMUS
Module: Gauss
Authors: Eugenio Aulisa, Giorgio Bornia, Erdi Kara

Copyright (c) FEMTTU
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "LineGaussLegendrePoints.hpp"
#include <iostream>
#include <stdlib.h>
#include <string.h>

namespace femus {
const unsigned line_gaussLegendre::GaussPoints[38] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19};

const double * line_gaussLegendre::Gauss[38] = {Gauss0[0], Gauss1[0], Gauss2[0], Gauss3[0], Gauss4[0], Gauss5[0], Gauss6[0], Gauss7[0], Gauss8[0], Gauss9[0], Gauss10[0], Gauss11[0], Gauss12[0], Gauss13[0], Gauss14[0], Gauss15[0], Gauss16[0], Gauss17[0], Gauss18[0], Gauss19[0], Gauss20[0], Gauss21[0], Gauss22[0], Gauss23[0], Gauss24[0], Gauss25[0], Gauss26[0], Gauss27[0], Gauss28[0], Gauss29[0], Gauss30[0], Gauss31[0], Gauss32[0], Gauss33[0], Gauss34[0], Gauss35[0], Gauss36[0], Gauss37[0]};

const double line_gaussLegendre::Gauss0[2][1] = {{2},
{0}};

const double line_gaussLegendre::Gauss1[2][1] = {{2},
{0}};

const double line_gaussLegendre::Gauss2[2][2] = {{1, 1},
{-0.57735026918963, 0.57735026918963}};

const double line_gaussLegendre::Gauss3[2][2] = {{1, 1},
{-0.57735026918963, 0.57735026918963}};

const double line_gaussLegendre::Gauss4[2][3] = {{0.55555555555556, 0.88888888888889, 0.55555555555556},
{-0.77459666924148, 0, 0.77459666924148}};

const double line_gaussLegendre::Gauss5[2][3] = {{0.55555555555556, 0.88888888888889, 0.55555555555556},
{-0.77459666924148, 0, 0.77459666924148}};

const double line_gaussLegendre::Gauss6[2][4] = {{0.34785484513745, 0.65214515486255, 0.65214515486255, 0.34785484513745},
{-0.86113631159405, -0.33998104358486, 0.33998104358486, 0.86113631159405}};

const double line_gaussLegendre::Gauss7[2][4] = {{0.34785484513745, 0.65214515486255, 0.65214515486255, 0.34785484513745},
{-0.86113631159405, -0.33998104358486, 0.33998104358486, 0.86113631159405}};

const double line_gaussLegendre::Gauss8[2][5] = {{0.23692688505619, 0.47862867049937, 0.56888888888889, 0.47862867049937, 0.23692688505619},
{-0.90617984593866, -0.53846931010568, 0, 0.53846931010568, 0.90617984593866}};

const double line_gaussLegendre::Gauss9[2][5] = {{0.23692688505619, 0.47862867049937, 0.56888888888889, 0.47862867049937, 0.23692688505619},
{-0.90617984593866, -0.53846931010568, 0, 0.53846931010568, 0.90617984593866}};

const double line_gaussLegendre::Gauss10[2][6] = {{0.36076157304814, 0.36076157304814, 0.46791393457269, 0.46791393457269, 0.17132449237917, 0.17132449237917},
{0.66120938646626, -0.66120938646626, -0.2386191860832, 0.2386191860832, -0.93246951420315, 0.93246951420315}};

const double line_gaussLegendre::Gauss11[2][6] = {{0.36076157304814, 0.36076157304814, 0.46791393457269, 0.46791393457269, 0.17132449237917, 0.17132449237917},
{0.66120938646626, -0.66120938646626, -0.2386191860832, 0.2386191860832, -0.93246951420315, 0.93246951420315}};

const double line_gaussLegendre::Gauss12[2][7] = {{0.41795918367347, 0.38183005050512, 0.27970539148928, 0.12948496616887, 0.38183005050512, 0.27970539148928, 0.12948496616887},
{0, 0.4058451513774, 0.74153118559939, 0.94910791234276, -0.4058451513774, -0.74153118559939, -0.94910791234276}};

const double line_gaussLegendre::Gauss13[2][7] = {{0.41795918367347, 0.38183005050512, 0.27970539148928, 0.12948496616887, 0.38183005050512, 0.27970539148928, 0.12948496616887},
{0, 0.4058451513774, 0.74153118559939, 0.94910791234276, -0.4058451513774, -0.74153118559939, -0.94910791234276}};

const double line_gaussLegendre::Gauss14[2][8] = {{0.36268378337836, 0.31370664587789, 0.22238103445338, 0.10122853629038, 0.36268378337836, 0.31370664587789, 0.22238103445338, 0.10122853629038},
{-0.18343464249565, -0.52553240991633, -0.79666647741363, -0.96028985649754, 0.18343464249565, 0.52553240991633, 0.79666647741363, 0.96028985649754}};

const double line_gaussLegendre::Gauss15[2][8] = {{0.36268378337836, 0.31370664587789, 0.22238103445338, 0.10122853629038, 0.36268378337836, 0.31370664587789, 0.22238103445338, 0.10122853629038},
{-0.18343464249565, -0.52553240991633, -0.79666647741363, -0.96028985649754, 0.18343464249565, 0.52553240991633, 0.79666647741363, 0.96028985649754}};

const double line_gaussLegendre::Gauss16[2][9] = {{0.33023935500126, 0.31234707704, 0.26061069640293, 0.18064816069486, 0.081274388361574, 0.31234707704, 0.26061069640293, 0.18064816069486, 0.081274388361574},
{0, -0.32425342340381, -0.61337143270059, -0.83603110732664, -0.96816023950763, 0.32425342340381, 0.61337143270059, 0.83603110732664, 0.96816023950763}};

const double line_gaussLegendre::Gauss17[2][9] = {{0.33023935500126, 0.31234707704, 0.26061069640293, 0.18064816069486, 0.081274388361574, 0.31234707704, 0.26061069640293, 0.18064816069486, 0.081274388361574},
{0, -0.32425342340381, -0.61337143270059, -0.83603110732664, -0.96816023950763, 0.32425342340381, 0.61337143270059, 0.83603110732664, 0.96816023950763}};

const double line_gaussLegendre::Gauss18[2][10] = {{0.29552422471475, 0.26926671931, 0.21908636251598, 0.14945134915058, 0.066671344308688, 0.29552422471475, 0.26926671931, 0.21908636251598, 0.14945134915058, 0.066671344308688},
{-0.14887433898163, -0.43339539412925, -0.67940956829902, -0.86506336668898, -0.97390652851717, 0.14887433898163, 0.43339539412925, 0.67940956829902, 0.86506336668898, 0.97390652851717}};

const double line_gaussLegendre::Gauss19[2][10] = {{0.29552422471475, 0.26926671931, 0.21908636251598, 0.14945134915058, 0.066671344308688, 0.29552422471475, 0.26926671931, 0.21908636251598, 0.14945134915058, 0.066671344308688},
{-0.14887433898163, -0.43339539412925, -0.67940956829902, -0.86506336668898, -0.97390652851717, 0.14887433898163, 0.43339539412925, 0.67940956829902, 0.86506336668898, 0.97390652851717}};

const double line_gaussLegendre::Gauss20[2][11] = {{0.2729250867779, 0.26280454451025, 0.23319376459199, 0.18629021092773, 0.12558036946491, 0.055668567116174, 0.26280454451025, 0.23319376459199, 0.18629021092773, 0.12558036946491, 0.055668567116174},
{0, -0.26954315595235, -0.51909612920681, -0.73015200557405, -0.88706259976809, -0.97822865814606, 0.26954315595235, 0.51909612920681, 0.73015200557405, 0.88706259976809, 0.97822865814606}};

const double line_gaussLegendre::Gauss21[2][11] = {{0.2729250867779, 0.26280454451025, 0.23319376459199, 0.18629021092773, 0.12558036946491, 0.055668567116174, 0.26280454451025, 0.23319376459199, 0.18629021092773, 0.12558036946491, 0.055668567116174},
{0, -0.26954315595235, -0.51909612920681, -0.73015200557405, -0.88706259976809, -0.97822865814606, 0.26954315595235, 0.51909612920681, 0.73015200557405, 0.88706259976809, 0.97822865814606}};

const double line_gaussLegendre::Gauss22[2][12] = {{0.2491470458134, 0.23349253653835, 0.20316742672307, 0.16007832854335, 0.10693932599532, 0.047175336386512, 0.2491470458134, 0.23349253653835, 0.20316742672307, 0.16007832854335, 0.10693932599532, 0.047175336386512},
{0.12523340851147, 0.36783149899818, 0.58731795428662, 0.76990267419431, 0.90411725637048, 0.98156063424672, -0.12523340851147, -0.36783149899818, -0.58731795428662, -0.76990267419431, -0.90411725637048, -0.98156063424672}};

const double line_gaussLegendre::Gauss23[2][12] = {{0.2491470458134, 0.23349253653835, 0.20316742672307, 0.16007832854335, 0.10693932599532, 0.047175336386512, 0.2491470458134, 0.23349253653835, 0.20316742672307, 0.16007832854335, 0.10693932599532, 0.047175336386512},
{0.12523340851147, 0.36783149899818, 0.58731795428662, 0.76990267419431, 0.90411725637048, 0.98156063424672, -0.12523340851147, -0.36783149899818, -0.58731795428662, -0.76990267419431, -0.90411725637048, -0.98156063424672}};

const double line_gaussLegendre::Gauss24[2][13] = {{0.23255155323087, 0.2262831802629, 0.20781604753689, 0.17814598076195, 0.13887351021979, 0.092121499837728, 0.040484004765316, 0.2262831802629, 0.20781604753689, 0.17814598076195, 0.13887351021979, 0.092121499837728, 0.040484004765316},
{0, 0.23045831595513, 0.44849275103645, 0.64234933944034, 0.80157809073331, 0.91759839922298, 0.98418305471859, -0.23045831595513, -0.44849275103645, -0.64234933944034, -0.80157809073331, -0.91759839922298, -0.98418305471859}};

const double line_gaussLegendre::Gauss25[2][13] = {{0.23255155323087, 0.2262831802629, 0.20781604753689, 0.17814598076195, 0.13887351021979, 0.092121499837728, 0.040484004765316, 0.2262831802629, 0.20781604753689, 0.17814598076195, 0.13887351021979, 0.092121499837728, 0.040484004765316},
{0, 0.23045831595513, 0.44849275103645, 0.64234933944034, 0.80157809073331, 0.91759839922298, 0.98418305471859, -0.23045831595513, -0.44849275103645, -0.64234933944034, -0.80157809073331, -0.91759839922298, -0.98418305471859}};

const double line_gaussLegendre::Gauss26[2][14] = {{0.21526385346316, 0.2051984637213, 0.18553839747794, 0.15720316715819, 0.1215185706879, 0.08015808715976, 0.035119460331752, 0.21526385346316, 0.2051984637213, 0.18553839747794, 0.15720316715819, 0.1215185706879, 0.08015808715976, 0.035119460331752},
{0.10805494870734, 0.31911236892789, 0.51524863635815, 0.68729290481169, 0.82720131506977, 0.92843488366357, 0.98628380869681, -0.10805494870734, -0.31911236892789, -0.51524863635815, -0.68729290481169, -0.82720131506977, -0.92843488366357, -0.98628380869681}};

const double line_gaussLegendre::Gauss27[2][14] = {{0.21526385346316, 0.2051984637213, 0.18553839747794, 0.15720316715819, 0.1215185706879, 0.08015808715976, 0.035119460331752, 0.21526385346316, 0.2051984637213, 0.18553839747794, 0.15720316715819, 0.1215185706879, 0.08015808715976, 0.035119460331752},
{0.10805494870734, 0.31911236892789, 0.51524863635815, 0.68729290481169, 0.82720131506977, 0.92843488366357, 0.98628380869681, -0.10805494870734, -0.31911236892789, -0.51524863635815, -0.68729290481169, -0.82720131506977, -0.92843488366357, -0.98628380869681}};

const double line_gaussLegendre::Gauss28[2][15] = {{0.20257824192556, 0.19843148532711, 0.18616100001556, 0.16626920581699, 0.13957067792615, 0.10715922046717, 0.070366047488108, 0.030753241996117, 0.19843148532711, 0.18616100001556, 0.16626920581699, 0.13957067792615, 0.10715922046717, 0.070366047488108, 0.030753241996117},
{0, 0.20119409399743, 0.39415134707756, 0.57097217260854, 0.72441773136017, 0.84820658341043, 0.93727339240071, 0.98799251802049, -0.20119409399743, -0.39415134707756, -0.57097217260854, -0.72441773136017, -0.84820658341043, -0.93727339240071, -0.98799251802049}};

const double line_gaussLegendre::Gauss29[2][15] = {{0.20257824192556, 0.19843148532711, 0.18616100001556, 0.16626920581699, 0.13957067792615, 0.10715922046717, 0.070366047488108, 0.030753241996117, 0.19843148532711, 0.18616100001556, 0.16626920581699, 0.13957067792615, 0.10715922046717, 0.070366047488108, 0.030753241996117},
{0, 0.20119409399743, 0.39415134707756, 0.57097217260854, 0.72441773136017, 0.84820658341043, 0.93727339240071, 0.98799251802049, -0.20119409399743, -0.39415134707756, -0.57097217260854, -0.72441773136017, -0.84820658341043, -0.93727339240071, -0.98799251802049}};

const double line_gaussLegendre::Gauss30[2][16] = {{0.18945061045507, 0.18260341504492, 0.169156519395, 0.14959598881658, 0.12462897125553, 0.095158511682492, 0.062253523938648, 0.027152459411754, 0.18945061045507, 0.18260341504492, 0.169156519395, 0.14959598881658, 0.12462897125553, 0.095158511682492, 0.062253523938648, 0.027152459411754},
{0.095012509837637, 0.28160355077926, 0.45801677765723, 0.61787624440264, 0.755404408355, 0.86563120238783, 0.94457502307323, 0.98940093499165, -0.095012509837637, -0.28160355077926, -0.45801677765723, -0.61787624440264, -0.755404408355, -0.86563120238783, -0.94457502307323, -0.98940093499165}};

const double line_gaussLegendre::Gauss31[2][16] = {{0.18945061045507, 0.18260341504492, 0.169156519395, 0.14959598881658, 0.12462897125553, 0.095158511682492, 0.062253523938648, 0.027152459411754, 0.18945061045507, 0.18260341504492, 0.169156519395, 0.14959598881658, 0.12462897125553, 0.095158511682492, 0.062253523938648, 0.027152459411754},
{0.095012509837637, 0.28160355077926, 0.45801677765723, 0.61787624440264, 0.755404408355, 0.86563120238783, 0.94457502307323, 0.98940093499165, -0.095012509837637, -0.28160355077926, -0.45801677765723, -0.61787624440264, -0.755404408355, -0.86563120238783, -0.94457502307323, -0.98940093499165}};

const double line_gaussLegendre::Gauss32[2][17] = {{0.17944647035621, 0.17656270536699, 0.16800410215645, 0.15404576107681, 0.13513636846853, 0.1118838471934, 0.085036148317179, 0.055459529373987, 0.024148302868548, 0.17656270536699, 0.16800410215645, 0.15404576107681, 0.13513636846853, 0.1118838471934, 0.085036148317179, 0.055459529373987, 0.024148302868548},
{0, 0.17848418149585, 0.35123176345388, 0.51269053708648, 0.65767115921669, 0.7815140038968, 0.88023915372699, 0.95067552176877, 0.99057547531442, -0.17848418149585, -0.35123176345388, -0.51269053708648, -0.65767115921669, -0.7815140038968, -0.88023915372699, -0.95067552176877, -0.99057547531442}};

const double line_gaussLegendre::Gauss33[2][17] = {{0.17944647035621, 0.17656270536699, 0.16800410215645, 0.15404576107681, 0.13513636846853, 0.1118838471934, 0.085036148317179, 0.055459529373987, 0.024148302868548, 0.17656270536699, 0.16800410215645, 0.15404576107681, 0.13513636846853, 0.1118838471934, 0.085036148317179, 0.055459529373987, 0.024148302868548},
{0, 0.17848418149585, 0.35123176345388, 0.51269053708648, 0.65767115921669, 0.7815140038968, 0.88023915372699, 0.95067552176877, 0.99057547531442, -0.17848418149585, -0.35123176345388, -0.51269053708648, -0.65767115921669, -0.7815140038968, -0.88023915372699, -0.95067552176877, -0.99057547531442}};

const double line_gaussLegendre::Gauss34[2][18] = {{0.16914238296314, 0.16427648374583, 0.15468467512626, 0.14064291467065, 0.12255520671148, 0.10094204410629, 0.076425730254889, 0.04971454889497, 0.021616013526483, 0.16914238296314, 0.16427648374583, 0.15468467512626, 0.14064291467065, 0.12255520671148, 0.10094204410629, 0.076425730254889, 0.04971454889497, 0.021616013526483},
{-0.084775013041735, -0.2518862256915, -0.41175116146284, -0.55977083107395, -0.69168704306035, -0.80370495897252, -0.89260246649756, -0.9558239495714, -0.99156516842093, 0.084775013041735, 0.2518862256915, 0.41175116146284, 0.55977083107395, 0.69168704306035, 0.80370495897252, 0.89260246649756, 0.9558239495714, 0.99156516842093}};

const double line_gaussLegendre::Gauss35[2][18] = {{0.16914238296314, 0.16427648374583, 0.15468467512626, 0.14064291467065, 0.12255520671148, 0.10094204410629, 0.076425730254889, 0.04971454889497, 0.021616013526483, 0.16914238296314, 0.16427648374583, 0.15468467512626, 0.14064291467065, 0.12255520671148, 0.10094204410629, 0.076425730254889, 0.04971454889497, 0.021616013526483},
{-0.084775013041735, -0.2518862256915, -0.41175116146284, -0.55977083107395, -0.69168704306035, -0.80370495897252, -0.89260246649756, -0.9558239495714, -0.99156516842093, 0.084775013041735, 0.2518862256915, 0.41175116146284, 0.55977083107395, 0.69168704306035, 0.80370495897252, 0.89260246649756, 0.9558239495714, 0.99156516842093}};

const double line_gaussLegendre::Gauss36[2][19] = {{0.16105444984878, 0.15896884339395, 0.15276604206586, 0.14260670217361, 0.12875396253934, 0.11156664554733, 0.09149002162245, 0.069044542737641, 0.0448142267657, 0.019461788229727, 0.15896884339395, 0.15276604206586, 0.14260670217361, 0.12875396253934, 0.11156664554733, 0.09149002162245, 0.069044542737641, 0.0448142267657, 0.019461788229727},
{0, -0.16035864564023, -0.31656409996363, -0.46457074137596, -0.60054530466168, -0.72096617733523, -0.82271465653714, -0.90315590361482, -0.96020815213483, -0.99240684384358, 0.16035864564023, 0.31656409996363, 0.46457074137596, 0.60054530466168, 0.72096617733523, 0.82271465653714, 0.90315590361482, 0.96020815213483, 0.99240684384358}};

const double line_gaussLegendre::Gauss37[2][19] = {{0.16105444984878, 0.15896884339395, 0.15276604206586, 0.14260670217361, 0.12875396253934, 0.11156664554733, 0.09149002162245, 0.069044542737641, 0.0448142267657, 0.019461788229727, 0.15896884339395, 0.15276604206586, 0.14260670217361, 0.12875396253934, 0.11156664554733, 0.09149002162245, 0.069044542737641, 0.0448142267657, 0.019461788229727},
{0, -0.16035864564023, -0.31656409996363, -0.46457074137596, -0.60054530466168, -0.72096617733523, -0.82271465653714, -0.90315590361482, -0.96020815213483, -0.99240684384358, 0.16035864564023, 0.31656409996363, 0.46457074137596, 0.60054530466168, 0.72096617733523, 0.82271465653714, 0.90315590361482, 0.96020815213483, 0.99240684384358}};

}