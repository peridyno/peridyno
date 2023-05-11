/**
 * Copyright 2021 Yue Chang
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
#include "Module/ConstraintModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	template<typename TDataType>
	class TriangularMeshConstraint : public ConstraintModule
	{
		DECLARE_TCLASS(TriangularMeshConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangularMeshConstraint();
		virtual ~TriangularMeshConstraint();

		void constrain() override;

	public:
		DEF_VAR(Real, Threshold, 0.0065, "Threshold for collision detection");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_ARRAY_IN(Triangle, TriangleIndex, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_IN(Coord, TriangleVertex, DeviceType::GPU, "triangle_vertex");

		DEF_ARRAYLIST_IN(int, TriangleNeighborIds, DeviceType::GPU, "triangle neighbors");

	protected:
		DArray<Coord> mPosBuffer;

		DArray<Coord> mPreviousPosition;
		DArray<Coord> mPrivousVertex;
	};
}
