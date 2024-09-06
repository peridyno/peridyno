/**
 * Copyright 2021~2024 Shusen Liu
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
#include "VirtualParticleGenerator.h"

#include "ParticleSystem/Module/Kernel.h"
#include "ParticleSystem/Module/SummationDensity.h"

#include "Collision/NeighborPointQuery.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*
	*@Brief: Spatially Adaptive Strategy (S.C.) in Dual-particle SPH method. (Virtual paritlce genorator).
	*@Note : When the strategy is adopted, the dual-particle method has the best quality
	*@Paper: Liu et al., ACM Trans Graph (TOG). 2024. (A Dual-Particle Approach for Incompressible SPH Fluids) doi.org/10.1145/3649888
	*/


	typedef unsigned short OcIndex;
	typedef unsigned long long int OcKey;
	typedef unsigned short Level;




	template<typename TDataType>
	class VirtualSpatiallyAdaptiveStrategy : public VirtualParticleGenerator<TDataType>
	{
		DECLARE_TCLASS(VirtualSpatiallyAdaptiveStrategy, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VirtualSpatiallyAdaptiveStrategy();
		~VirtualSpatiallyAdaptiveStrategy() override;

		void constrain() override;

		void setHashGridSize(Real x) { gridSize = x; };

		DECLARE_ENUM(CandidatePointCount,
			neighbors_8 = 8,
			neighbors_27 = 27,
			neighbors_33 = 33,
			neighbors_125 = 125
		);

		DEF_ENUM(CandidatePointCount, CandidatePointCount, CandidatePointCount::neighbors_33, "Candidate Point Count");


		DEF_VAR(Real, RestDensity, Real(1000), "Reference density");


		/**
		* @brief Virtual Particles Sampling Distance
		*/

		DEF_VAR(Real, SamplingDistance, Real(0.005), "");

		/**
		* @brief Real Particle positions
		*/
		DEF_ARRAY_IN(Coord, RPosition, DeviceType::GPU, "Input real particle position");


	private:
		SpikyKernel<Real> m_kernel;

		Real gridSize;


		DArray<Coord> m_anchorPoint;

		Coord origin = Coord(0.0f, 0.0f, 0.0f);

		/*
		* @brief Morton codes of anchor points.
		*/
		DArray<uint32_t> m_anchorPointCodes;

		DArray<uint32_t> m_nonRepeatedCount;

		DArray<uint32_t> m_candidateCodes;

		DArray<Coord> m_virtual_position;

	private:
		//std::shared_ptr<SummationDensity<TDataType>> m_v_summation;

		std::shared_ptr <SummationDensity<TDataType>> m_vv_density;


		std::shared_ptr<NeighborPointQuery<TDataType>> m_rv_nbrQuery;
		std::shared_ptr<NeighborPointQuery<TDataType>> m_vr_nbrQuery;
		std::shared_ptr<NeighborPointQuery<TDataType>> m_vv_nbrQuery;

		
	};
}