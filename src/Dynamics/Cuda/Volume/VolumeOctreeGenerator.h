/**
 * Copyright 2022 Lixin Ren
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

#include "Topology/TriangleSet.h"

namespace dyno 
{
	/**
	 * @brief This is a GPU-based implementation of algebraic adaptive signed distance field (AASDF).
	 * 		  For more details, please refer to "Algebraic Adaptive Signed Distance Field on GPU" by [Ren et.al. 2022].
	 */
	template<typename TDataType>
	class VolumeOctreeGenerator :public VolumeOctree<TDataType>
	{
		DECLARE_TCLASS(VolumeOctreeGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeOctreeGenerator();
		~VolumeOctreeGenerator() override;

		void load(std::string filename);

		void load(std::string filename, Coord rotate_value, Real scale_value, Coord translate_value);

		Coord lowerBound() override { return m_origin; }
		Coord upperBound() override { return m_origin + this->varSpacing()->getData() * Coord(m_nx, m_ny, m_nz); }

	public:

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "The triangles of closed surface");
		
		DEF_VAR(Real, Spacing, Real(0.1), "the dx");

		DEF_VAR(uint, Padding, 10, "Padding of model");

		DEF_VAR(int, AABBPadding, 1, "Padding of each triangle`s AABB");

		DEF_VAR(Coord, ForwardVector, Coord(0), "The distance and direction of topology move");

	protected:
		void resetStates() override;

		void updateStates() override;

		void updateTopology() override;

	private:
		void initParameter();

		int m_nx;
		int m_ny;
		int m_nz;
		Coord m_origin;

		int m_level0 = 0;
		int m_max_grid = 50000;
	};
}
