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
#include "Samplers/Sampler.h"

namespace dyno
{
	template<typename TDataType>
	class UpperSemiCircleSampler : public Sampler<TDataType>
	{
		DECLARE_TCLASS(UpperSemiCircleSampler, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UpperSemiCircleSampler();

	public:
		DEF_VAR(Real, SamplingDistance, 0.004, "Sampling distance");

		DEF_VAR(Coord, Center, Vec3f(0.0f, 0.3f, 0.0f), "The center of the circle");
		DEF_VAR(Real, Radius, 0.3, "The radius of the circle");
		DEF_VAR(Real, Dx, 0.008, "The radius of the circle");
		DEF_VAR(Real, YPlane, 0.3, "");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(UpperSemiCircleSampler, TDataType);
}