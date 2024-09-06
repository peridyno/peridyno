/**
 * Copyright 2024 Shusen Liu
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
#include "Topology/TriangleSet.h"

namespace dyno {

	/*
	*	\brief	Particle volume force along normal direction of neighbor mesh.
	*/

	template<typename TDataType>
	class NormalForce : public ConstraintModule
	{

		DECLARE_TCLASS(NormalForce, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NormalForce();
		~NormalForce() override ;

		DEF_ARRAY_OUT(Coord, NormalForce, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, ParticleNormal, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, ""); 

		DEF_VAR(Real, Strength, 10.0f, "");

		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAYLIST_IN(int, TriangleNeighborIds, DeviceType::GPU, "triangle neighbors");

		DEF_ARRAY_IN(int, ParticleMeshID, DeviceType::GPU, "triangle neighbors");


		void constrain() override;

	private:
		DArray<bool> mNormalForceFlag;

	};

	IMPLEMENT_TCLASS(NormalForce, TDataType)
}