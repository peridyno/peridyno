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

namespace dyno {

	template<typename TDataType> class SummationDensity;


	/*
	*@Brief: Colocaition strategy in Dual-particle SPH method. (Virtual paritlce genorator).
	*@Note : If the strategy is used, the Dual-particle SPH method will be degenerate to standard ISPH method (Cummins et al., An SPH Projection Method.Journal of Computational Physics 152)
	*@Paper: Liu et al., ACM Trans Graph (TOG). 2024. (A Dual-Particle Approach for Incompressible SPH Fluids) doi.org/10.1145/3649888
	*/

	template<typename TDataType>
	class VirtualColocationStrategy : public VirtualParticleGenerator<TDataType>
	{

		DECLARE_TCLASS(VirtualColocationStrategy, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VirtualColocationStrategy();
		~VirtualColocationStrategy() override;

		void constrain() override;

		DEF_ARRAY_IN(Coord, RPosition, DeviceType::GPU, "Input real particle position");
	};
}