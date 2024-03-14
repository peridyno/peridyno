/**
 * Copyright 2017-2021 Xiaowei He
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

#include "Algorithm/Reduction.h"

namespace dyno 
{
	template<typename TDataType>
	class CalculateMinimum : public ComputeModule
	{
		DECLARE_TCLASS(CalculateMinimum, TDataType)
	public:
		typedef typename TDataType::Real Real;

		CalculateMinimum() {};
		~CalculateMinimum() override {};

		void compute() override;

	public:
		DEF_ARRAY_IN(Real, ScalarArray, DeviceType::GPU, "");
		DEF_VAR_OUT(Real, Scalar, "");

	private:
		Reduction<Real> mReduce;
	};
}
