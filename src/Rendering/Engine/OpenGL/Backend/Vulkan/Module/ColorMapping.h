/**
 * Copyright 2017-2023 Xiaowei HE
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
	class ColorMapping : public ComputeModule
	{
		DECLARE_CLASS(ColorMapping)
	public:
		DECLARE_ENUM(ColorTable,
			Jet = 0,
			Heat = 1);

		ColorMapping();
		~ColorMapping() override {};

		void compute() override;

	public:
		DEF_ENUM(ColorTable, Type, ColorTable::Jet, "");

		DEF_VAR(float, Min, 0.0f, "");
		DEF_VAR(float, Max, 1.0f, "");

		DEF_ARRAY_IN(float, Scalar, DeviceType::GPU, "");
		DEF_ARRAY_OUT(Vec3f, Color, DeviceType::GPU, "");
	};
}
