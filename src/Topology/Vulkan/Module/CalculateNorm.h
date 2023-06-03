/**
 * Copyright 2017-2023 Xiaowei He
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
	class CalculateNorm : public ComputeModule
	{
		DECLARE_CLASS(ColorMapping)
	public:
		CalculateNorm();
		~CalculateNorm() override {};

		void compute() override;

	public:
		DEF_ARRAY_IN(Vec3f, Vec, DeviceType::GPU, "");
		DEF_ARRAY_OUT(float, Norm, DeviceType::GPU, "");
	};
}
