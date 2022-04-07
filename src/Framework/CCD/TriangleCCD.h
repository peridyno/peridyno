/**
 * Copyright 2021 Xiaowei He
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
#include "Topology/Primitive3D.h"
#include "TightCCD.h"

namespace dyno 
{
	template<typename Real>
	DYN_FUNC void TriangleCCD(
		TTriangle3D<Real>& s0, TTriangle3D<Real>& s1,
		TTriangle3D<Real>& t0, TTriangle3D<Real>& t1,
		Real& toi)
	{
		Vector<Real, 3> p[3];
		p[0] = s0.v[0];
		p[1] = s0.v[1];
		p[2] = s0.v[2];

		Vector<Real, 3> pp[3];
		pp[0] = s1.v[0];
		pp[1] = s1.v[1];
		pp[2] = s1.v[2];

		Vector<Real, 3> q[3];
		q[0] = t0.v[0];
		q[1] = t0.v[1];
		q[2] = t0.v[2];

		Vector<Real, 3> qq[3];
		qq[0] = t1.v[0];
		qq[1] = t1.v[1];
		qq[2] = t1.v[2];

		Real toi = 2;

		///*
		//VF
		Real t;
		for (int st = 0; st < 3; st++)
		{
			bool collided = VertexFaceCCD(
				p[st], q[0], q[1], q[2],
				pp[st], qq[0], qq[1], qq[2],
				t);

			toi = collided ? min(t, toi) : toi;
		}

		//VF
		for (int st = 0; st < 3; st++)
		{
			bool collided = VertexFaceCCD(q[st], p[0], p[1], p[2],
				qq[st], pp[0], pp[1], pp[2],
				t);
			toi = collided ? min(t, toi) : toi;
		}

		//EE
		for (int st = 0; st < 3; st++)
		{
			int ind0 = st;
			int ind1 = (st + 1) % 3;
			for (int ss = 0; ss < 3; ss++)
			{
				int ind2 = st;
				int ind3 = (st + 1) % 3;
				bool collided = EdgeEdgeCCD(
					p[ind0], p[ind1], q[ind2], q[ind3],
					pp[ind0], pp[ind1], qq[ind2], qq[ind3],
					t);
				toi = collided ? min(t, toi) : toi;
			}
		}
	}
}
