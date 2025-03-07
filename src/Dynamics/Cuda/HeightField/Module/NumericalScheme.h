/**
 * Copyright 2024 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "Vector/Vector4D.h"

namespace dyno 
{
#define POSITIVTY Real(0.1)

	//A stable solution to calculate u and v
	template<typename Real, typename Coord4D>
	inline DYN_FUNC void ComputeVelocity(Real& u, Real& v, Coord4D gp)
	{
		Real EPSILON = 0.000001f;

		Real h = maximum(gp.x, 0.0f);
		Real h4 = h * h * h * h;
		u = sqrtf(2.0f) * h * gp.y / (sqrtf(h4 + maximum(h4, EPSILON)));
		v = sqrtf(2.0f) * h * gp.z / (sqrtf(h4 + maximum(h4, EPSILON)));
	}

	//A first-order upwind scheme to calculate -ghB'
	template<typename Real, typename Coord4D>
	DYN_FUNC Real FirstOrderUpwindPotential(Coord4D gpl, Coord4D gpr, Real GRAVITY, Real dt)
	{
		Real EPSILON = 0.000001f;

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real ul = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + maximum(hl4, EPSILON)));
		Real vl = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + maximum(hl4, EPSILON)));

		Real hr = maximum(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real ur = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + maximum(hr4, EPSILON)));
		Real vr = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + maximum(hr4, EPSILON)));

		//Potential
		Real wl = hl + gpl.w;
		Real wr = hr + gpr.w;
		Real potential = 0;
		//Left
		{
			Real wl_diff = maximum(wl - gpr.w, Real(0));
			potential -= GRAVITY * hl * wl_diff;
		}
		// Right
		{
			Real wr_diff = maximum(wr - gpl.w, Real(0));
			potential += GRAVITY * hr * wr_diff;
		}

		return - dt * potential;
	}

	//A first-order upwind scheme to calculate (hu, huu, huv, 0)
	template<typename Real, typename Coord4D>
	DYN_FUNC Coord4D FirstOrderUpwindX(Coord4D gpl, Coord4D gpr, Real GRAVITY, Real dt)
	{
		Real EPSILON = 0.000001f;

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real ul = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + maximum(hl4, EPSILON)));
		Real vl = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + maximum(hl4, EPSILON)));

		Real hr = maximum(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real ur = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + maximum(hr4, EPSILON)));
		Real vr = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + maximum(hr4, EPSILON)));

		Real hu = 0.5 * (hl * ul + hr * ur);
		Real hm = 0.5f * (hl + hr);
		Real hm4 = hm * hm * hm * hm;
		Real um = sqrtf(2.0f) * hm * hu / (sqrtf(hm4 + maximum(hm4, EPSILON)));
		Real vm = 0.5f * (vl + vr);

		//Advection
		Real h_hat = um >= 0 ? maximum(hl - maximum(gpr.w - gpl.w, 0.0f), 0.0f) : maximum(hr - maximum(gpl.w - gpr.w, 0.0f), 0.0f);
		Real uh_hat = h_hat * um;// -dt * potential;
		Real vh_hat = h_hat * vm;

		Real alpha = (um >= 0 ? hl : hr) / maximum(dt * uh_hat, EPSILON);
		Coord4D flux = minimum(alpha, POSITIVTY) * Coord4D(uh_hat, uh_hat * um, vh_hat * um, 0.0f);

		return flux;
	}

	//A first-order upwind scheme to calculate (hv, huv, hvv, 0)
	template<typename Real, typename Coord4D>
	DYN_FUNC Coord4D FirstOrderUpwindY(Coord4D gpl, Coord4D gpr, Real GRAVITY, Real dt)
	{
		Real EPSILON = 0.000001f;

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real ul = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + maximum(hl4, EPSILON)));
		Real vl = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + maximum(hl4, EPSILON)));

		Real hr = maximum(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real ur = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + maximum(hr4, EPSILON)));
		Real vr = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + maximum(hr4, EPSILON)));

		Real hv = 0.5 * (hl * vl + hr * vr);
		Real hm = 0.5f * (hl + hr);
		Real hm4 = hm * hm * hm * hm;
		Real um = 0.5f * (ul + ur);
		Real vm = sqrtf(2.0f) * hm * hv / (sqrtf(hm4 + maximum(hm4, EPSILON))); 

		//Advection
		Real h_hat = vm >= 0 ? maximum(hl - maximum(gpr.w - gpl.w, 0.0f), 0.0f) : maximum(hr - maximum(gpl.w - gpr.w, 0.0f), 0.0f);
		Real uh_hat = h_hat * um;
		Real vh_hat = h_hat * vm;// -dt * potential;

		Real alpha = (vm >= 0 ? hl : hr) / maximum(dt * vh_hat, EPSILON);
		Coord4D flux = minimum(alpha, POSITIVTY) * Coord4D(vh_hat, uh_hat * vm, vh_hat * vm, 0.0f);
		
		return flux;
	}

	template<typename Real, typename Coord4D>
	DYN_FUNC Coord4D CentralUpwindX(Coord4D gpl, Coord4D gpr, Real GRAVITY)
	{
		Real EPSILON = 0.00001f;

		Real h = maximum(0.5f * (gpl.x + gpr.x), 0.0f);
		Real b = 0.5f * (gpl.w + gpr.w);

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real ul = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + maximum(hl4, Real(EPSILON))));

		Real hr = maximum(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real ur = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + maximum(hr4, Real(EPSILON))));

		if (hl < EPSILON && hr < EPSILON)
		{
			return Coord4D(0.0f);
		}

		Real a_plus;
		Real a_minus;
		a_plus = maximum(maximum(Real(ul + sqrtf(GRAVITY * (gpl.x/*+gpl.w*/))), Real(ur + sqrtf(GRAVITY * (gpr.x/*+gpr.w*/)))), Real(0));
		a_minus = minimum(minimum(Real(ul - sqrtf(GRAVITY * (gpl.x/*+gpl.w*/))), Real(ur - sqrtf(GRAVITY * (gpr.x/*+gpr.w*/)))), Real(0));

		Coord4D delta_U = gpr - gpl;
		if (gpl.x > EPSILON && gpr.x > EPSILON) {
			delta_U.x += delta_U.w;
		}

		delta_U.w = 0.0f;

		Coord4D Fl = Coord4D(gpl.y, gpl.y * ul, gpl.z * ul, 0.0f);
		Coord4D Fr = Coord4D(gpr.y, gpr.y * ur, gpr.z * ur, 0.0f);

		Coord4D re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus + EPSILON) + a_plus * a_minus / (a_plus - a_minus + EPSILON) * delta_U;

		if (ul == 0 && ur == 0) {//abs(ul) <EPSILON && abs(ur) <EPSILON
			re.x = 0;
			re.y = 0;
			re.z = 0;
		}

		return re;
	}

	template<typename Real, typename Coord4D>
	DYN_FUNC Coord4D CentralUpwindY(Coord4D gpl, Coord4D gpr, Real GRAVITY)
	{
		Real EPSILON = 0.00001f;

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real vl = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + max(hl4, EPSILON)));

		Real hr = maximum(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real vr = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + max(hr4, EPSILON)));

		if (hl < EPSILON && hr < EPSILON)
		{
			return Coord4D(0.0f);
		}

		Real a_plus = maximum(maximum(Real(vl + sqrtf(GRAVITY * (gpl.x/* + gpl.w*/))), Real(vr + sqrtf(GRAVITY * (gpr.x/* + gpr.w*/)))), Real(0));
		Real a_minus = minimum(minimum(Real(vl - sqrtf(GRAVITY * (gpl.x/* + gpl.w*/))), Real(vr - sqrtf(GRAVITY * (gpr.x/* + gpr.w*/)))), Real(0));

		Real b = 0.5f * (gpl.w + gpr.w);

		Coord4D delta_U = gpr - gpl;
		if (gpl.x > EPSILON && gpr.x > EPSILON)
		{
			delta_U.x += delta_U.w;
		}
		delta_U.w = 0.0f;

		Coord4D Fl = Coord4D(gpl.z, gpl.y * vl, gpl.z * vl, 0.0f);
		Coord4D Fr = Coord4D(gpr.z, gpr.y * vr, gpr.z * vr, 0.0f);

		Coord4D re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus) + a_plus * a_minus / (a_plus - a_minus) * delta_U;

		if (vl == 0 && vr == 0)
		{
			re.x = 0;
			re.y = 0;
			re.z = 0;
		}
		return re;
	}
}
