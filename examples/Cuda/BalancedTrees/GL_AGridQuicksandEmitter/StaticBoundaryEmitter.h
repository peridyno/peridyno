/**
 * Copyright 2024 Lixin Ren
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
#include "Module/ComputeModule.h"

namespace dyno 
{

	template<typename TDataType>
	class StaticBoundaryEmitter : public ComputeModule
	{
		DECLARE_TCLASS(StaticBoundaryEmitter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticBoundaryEmitter() {};
		~StaticBoundaryEmitter() override {};

	protected:
		void compute() override;

	public:
		DEF_VAR(Coord, Location, Vec3f(0.0f, 0.3f, 0.0f), "The location of emitter");
		DEF_VAR(Real, XWidth, 0.03, "");
		DEF_VAR(Real, YHigh, 0.03, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
	};
}
