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
	class StaticBoundaryCircle : public ComputeModule
	{
		DECLARE_TCLASS(StaticBoundaryCircle, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticBoundaryCircle() {};
		~StaticBoundaryCircle() override {};

	protected:
		void compute() override;

	public:
		DEF_VAR(Real, PlaneTangentialFriction, 0.0, "Tangential friction of XY plane");
		DEF_VAR(Real, TangentialFriction, 0.0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 1.0, "Normal friction");

		DEF_VAR(Coord, Center, Vec3f(0.0f, 0.3f, 0.0f), "The center of the circle");
		DEF_VAR(Real, Radius, 0.3, "The radius of the circle");
		DEF_VAR(Real, Dx, 0.008, "The radius of the circle");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

	};
}
