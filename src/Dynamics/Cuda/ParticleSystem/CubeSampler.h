/**
 * Copyright 2022 Xiaowei He
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
	class CubeSampler : public Sampler<TDataType>
	{
		DECLARE_TCLASS(CubeSampler, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CubeSampler();

	public:
		DEF_VAR(Real, SamplingDistance, 0.1, "Sampling distance");

		DEF_VAR_IN(TOrientedBox3D<Real>, Cube,  "");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(CubeSampler, TDataType);
}