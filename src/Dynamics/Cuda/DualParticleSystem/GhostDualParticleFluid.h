/**
 * Copyright 2025 Shusen Liu
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
#include "DualParticleFluid.h"
#include "ParticleSystem/GhostFluid.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/Emitters/ParticleEmitter.h"
#include "Topology/PointSet.h"
#include "Collision/Attribute.h"
#include "Module/VirtualSpatiallyAdaptiveStrategy.h"
#include "Module/VirtualColocationStrategy.h"
#include "Module/VirtualParticleShiftingStrategy.h"

namespace dyno
{


	template<typename TDataType>
	class GhostDualParticleFluid :  public DualParticleFluid<TDataType>
	{
		DECLARE_TCLASS(GhostDualParticleFluid, TDataType)

	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GhostDualParticleFluid();
		GhostDualParticleFluid(int key);

		~GhostDualParticleFluid();

		/**
		 * @brief Particle position for both the fluid and solid
		 */
		DEF_ARRAY_STATE(Coord, PositionMerged, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_ARRAY_STATE(Coord, VelocityMerged, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle Attribute
		 */
		DEF_ARRAY_STATE(Attribute, AttributeMerged, DeviceType::GPU, "Particle attribute");


		/**
		* @brief Particle Normal
		*/
		DEF_ARRAY_STATE(Coord, NormalMerged, DeviceType::GPU, "Particle normal");


		DEF_NODE_PORTS(GhostParticles<TDataType>, BoundaryParticle, "Initial boundary ghost particles");


	protected:

		void resetStates();

		void preUpdateStates();

		void postUpdateStates();

	private:

		void constructMergedArrays();
	};

	IMPLEMENT_TCLASS(GhostDualParticleFluid, TDataType)
}