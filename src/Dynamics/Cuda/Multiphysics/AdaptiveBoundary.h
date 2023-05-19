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

#include "Volume/VolumeOctree.h"

#include "ParticleSystem/ParticleSystem.h"

#include "Peridynamics/TriangularSystem.h"

#include "RigidBody/RigidBody.h"

namespace dyno 
{
	template<typename TDataType>
	class AdaptiveBoundary : public Node
	{
		DECLARE_TCLASS(AdaptiveBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveBoundary();
		~AdaptiveBoundary() override;

	public:
		DEF_VAR(Real, TangentialFriction, 0.0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 1.0, "Normal friction");

		DEF_NODE_PORTS(RigidBody<TDataType>, RigidBody, "A rigid body");


		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");

		DEF_NODE_PORTS(TriangularSystem<TDataType>, TriangularSystem, "Triangular Systems");

		DEF_NODE_PORTS(VolumeOctree<TDataType>, Boundary, "Adaptive SDF for Obstacles");

	protected:
		void updateStates() override;

	private:
		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
	};
}
