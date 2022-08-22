/**
 * Copyright 2022 Xiaowei He & Shusen Liu
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
#include "Sampler.h"

namespace dyno
{
	template<typename TDataType>
	class SphereSampler : public Sampler<TDataType>
	{
		DECLARE_TCLASS(SphereSampler, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SphereSampler();

	public:
		DEF_VAR(Real, SamplingDistance, 0.05, "Sampling distance");

		DEF_VAR_IN(TSphere3D<Real>, Sphere,  "");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(SphereSampler, TDataType);
}