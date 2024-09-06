/**
 * Copyright 2017-2021 Lixin Ren
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

#include "VolumeOctree.h"

#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"
#include "Vector.h"


namespace dyno 
{
	class UniformNode
	{
	public:

		DYN_FUNC bool operator< (const UniformNode& ug) const
		{
			return position_index < ug.position_index;
		}

		DYN_FUNC void set_value(int surf, int pos)
		{
			surface_index = surf;
			position_index = pos;
		}

		int surface_index = -1;
		int position_index = -1;
	};

	struct NodeCmp
	{
		DYN_FUNC bool operator()(const UniformNode& A, const UniformNode& B)
		{
			return A < B;
		}
	};

	template<typename TDataType>
	class VolumeUniformGenerator :public VolumeOctree<TDataType>
	{
		DECLARE_TCLASS(VolumeUniformGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;


		VolumeUniformGenerator();
		~VolumeUniformGenerator() override;

		void load(std::string filename);

		void resetStates() override;
		void updateStates() override;

		void initParameter();

		DYN_FUNC inline Real Dx() const { return m_dx; }
		DYN_FUNC inline Coord Origin() const { return m_origin; }
		DYN_FUNC inline int nx() const { return m_nx; }
		DYN_FUNC inline int ny() const { return m_ny; }
		DYN_FUNC inline int nz() const { return m_nz; }

		void getSignDistance(DArray<Coord> point_pos, DArray<Real>& point_sdf);

	private:

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "The triangles of closed surface");

		DEF_VAR(Real, Spacing, 1.0, "the dx");

		DEF_VAR(uint, Padding, 1, "");

		DEF_VAR(Coord, AnchorOrigin, 0, "Anchor origin");

		DEF_VAR(Coord, ForwardVector, 0, "The distance and direction of topology move");

		DEF_VAR_OUT(Coord, UniformOrigin, "Uniform grids origin");
		DEF_VAR_OUT(uint, Unx, "");
		DEF_VAR_OUT(uint, Uny, "");
		DEF_VAR_OUT(uint, Unz, "");

	private:

		Coord m_origin;
		Real m_dx;
		int m_nx;
		int m_ny;
		int m_nz;
	};
}
