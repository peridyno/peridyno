/**
 * Copyright 2021 Nurshat Menglik
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
#include "Matrix/Transform3x3.h"
#include "Module/ComputeModule.h"

namespace dyno 
{
	template<typename TDataType>
	class ComputeParticleAnisotropy : public ComputeModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename Transform<Real, 3> Transform;

		ComputeParticleAnisotropy();
		~ComputeParticleAnisotropy();

		void compute() override;
		
	public:
		DEF_VAR(Real, SmoothingLength, Real(0.0125), "smoothing length");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Transform, Transform, DeviceType::GPU, "transform");
	};
}