/**
 * Copyright 2024 Xiaowei He
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
#include "Node.h"

#include "ParticleSystem/ParticleSystem.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	/**
	 * @brief A triangular mesh boundary to prevent interpenetration for particles
	 */
	template<typename TDataType>
	class TriangularMeshBoundary : public Node
	{
		DECLARE_TCLASS(TriangularMeshBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TriangularMeshBoundary();
		~TriangularMeshBoundary() override;


	public:
		DEF_VAR(Real, Thickness, 0.0065, "Mesh thickness used for collision detection");

		DEF_VAR(Real, TangentialFriction, 0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 0, "Normal friction");

	public:
		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

	public:
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Particle velocity");

	protected:
		void preUpdateStates() override;
		void updateStates() override;
		void postUpdateStates() override;

	private:
	};
}
