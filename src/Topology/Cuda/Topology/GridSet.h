/**
 * Copyright 2021 Lixin Ren
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
#include "Module/TopologyModule.h"

namespace dyno
{
	template<typename TDataType>
	class GridSet : public TopologyModule
	{
		DECLARE_TCLASS(GridSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GridSet();
		~GridSet();

		void setUniGrid(int ni, int nj, int nk, Real dxmm, Coord lo_center);
		void setNijk(int ni, int nj, int nk);
		void setOrigin(Coord lo_center) { m_origin = lo_center; }
		void setDx(Real dxmm) { m_dx = dxmm; }

		int getNi() { return m_ni; }
		int getNj() { return m_nj; }
		int getNk() { return m_nk; }
		Coord getOrigin() { return m_origin; }
		Real getDx() { return m_dx; }
	private:
		int m_ni, m_nj, m_nk;
		Real m_dx;
		Coord m_origin;
	};
}


