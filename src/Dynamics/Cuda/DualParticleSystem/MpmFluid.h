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

#pragma once
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/Emitters/ParticleEmitter.h"
#include "Topology/PointSet.h"
#include "Collision/Attribute.h"
#include "Module/VirtualSpatiallyAdaptiveStrategy.h"
#include "Module/VirtualColocationStrategy.h"
#include "Module/VirtualParticleShiftingStrategy.h"
#include "Module/FlipFluidExplicitSolver.h"

namespace dyno
{
	template<typename TDataType>
	class MpmFluid : public ParticleFluid<TDataType>
	{
		DECLARE_TCLASS(MpmFluid, TDataType)

	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MpmFluid();
		~MpmFluid();

		DEF_ARRAY_STATE(Attribute, ParticleAttribute, DeviceType::GPU, "Real Particle Attribute");

		DEF_ARRAY_STATE(Coord, BoundaryNorm, DeviceType::GPU, "Boundary Norm");

		DEF_ARRAY_STATE(Coord, GridPosition, DeviceType::GPU, "Grid Position");

		DEF_INSTANCE_STATE(PointSet<TDataType>, VirtualPointSet, "Topology");

		DEF_ARRAY_STATE(Coord, GridVelocity, DeviceType::GPU, "Velocity on Grid");

		DEF_VAR_STATE(Real, GridSpacing, 0.005, "Spacing distance of grids");

	protected:

		void resetStates();

		void preUpdateStates();

		void postUpdateStates();

		std::shared_ptr<VirtualParticleGenerator<TDataType>> vpGen;

	private:

	};

	IMPLEMENT_TCLASS(MpmFluid, TDataType)
}

