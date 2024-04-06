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
	*@Brief: Particle shifting strategy in Dual-particle SPH method. (Virtual paritlce genorator).
	*@Note : Implementation is based on  PBF method (Position Base Fluid.)
	*@Paper: Liu et al., ACM Trans Graph (TOG). 2024. (A Dual-Particle Approach for Incompressible SPH Fluids) doi.org/10.1145/3649888
	*/

	template<typename TDataType>
	class VirtualParticleShiftingStrategy : public VirtualParticleGenerator<TDataType>
	{

		DECLARE_TCLASS(VirtualParticleShiftingStrategy, TDataType)


	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VirtualParticleShiftingStrategy();
		~VirtualParticleShiftingStrategy() override;

		void constrain() override;

		void takeOneIteration();

		bool VectorResize();

		//public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, Real(1000), "Reference density");

		DEF_VAR(Real, SamplingDistance, Real(0.005), "");

		DEF_VAR(Real, SmoothingLength, Real(0.0075), "");

		//DEF_VAR(Real, VirtualRestDensity, Real(1000.0f), "");

		DEF_VAR_IN(uint, FrameNumber, "Frame number");


		/**
		* @brief Real Particle positions
		*/
		DEF_ARRAY_IN(Coord, RPosition, DeviceType::GPU, "Input real particle position");

		/*
		*@brief Virtual neighbors of Virtual Particles
		*/
		DEF_ARRAYLIST_OUT(int, VVNeighborIds, DeviceType::GPU, "Return virtual particles' virtual neighbor ids");


		/**
		* @brief Final particle densities
		*/
		DEF_ARRAY_OUT(Real, VDensity, DeviceType::GPU, "Final particle density");

		//DEF_EMPTY_IN_ARRAY(Type, ParticleType, DeviceType::GPU, "Particle Type");

	private:
		SpikyKernel<Real> m_kernel;
		DArray<Real> m_lamda;
		DArray<Coord> m_deltaPos;
		//	DArray<Coord> m_position_old;

		Real maxDensity;

	private:
		//std::shared_ptr<SummationDensity<TDataType>> m_v_summation;

		std::shared_ptr <SummationDensity<TDataType>> m_vv_density;
		std::shared_ptr <NeighborPointQuery<TDataType>> m_vv_nbrQuery;
	};
}