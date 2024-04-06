/**
 * Copyright 2022-2024 Shusen Liu
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

//#pragma once
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/ParticleEmitter.h"
#include "Topology/PointSet.h"
#include "Collision/Attribute.h"
#include "VirtualSpatiallyAdaptiveStrategy.h"
#include "VirtualColocationStrategy.h"
#include "VirtualParticleShiftingStrategy.h"

namespace dyno
{
	template<typename TDataType>
	class DualParticleFluidSystem : public ParticleFluid<TDataType>
	{
		DECLARE_TCLASS(DualParticleFluidSystem, TDataType)

	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DualParticleFluidSystem();
		DualParticleFluidSystem(int key);
		~DualParticleFluidSystem();

		DEF_ARRAY_STATE(Attribute, ParticleAttribute, DeviceType::GPU, "Real Particle Attribute");

		DEF_ARRAY_STATE(Coord, BoundaryNorm, DeviceType::GPU, "Boundary Norm");

		DEF_ARRAY_STATE(Coord, VirtualPosition, DeviceType::GPU, "Virtual Particle");

		DEF_INSTANCE_STATE(PointSet<TDataType>, VirtualPointSet, "Topology");

		DECLARE_ENUM(EVirtualParticleSamplingStrategy,
		ColocationStrategy = 0,
		ParticleShiftingStrategy = 1,
		SpatiallyAdaptiveStrategy = 2);

		DEF_ENUM(EVirtualParticleSamplingStrategy, 
			VirtualParticleSamplingStrategy, 
			EVirtualParticleSamplingStrategy::SpatiallyAdaptiveStrategy,
			"Virtual Particle Sampling Strategy");

	protected:

		void resetStates();

		void preUpdateStates();

		void postUpdateStates();

		void animationPipelineWithoutVirtualPartilce(
			int key
		);

		std::shared_ptr<VirtualParticleGenerator<TDataType>> vpGen;

	private:

	};

	IMPLEMENT_TCLASS(DualParticleFluidSystem, TDataType)
}

