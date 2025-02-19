/**
 * Copyright 2024 Yuzhong Guo
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
#include "ArticulatedBody.h"

#include "STL/Pair.h"

#include "Field/VehicleInfo.h"

namespace dyno
{
	template<typename TDataType>
	class ConfigurableBody : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(ConfigurableBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ConfigurableBody();
		~ConfigurableBody() override;

		DEF_VAR(VehicleBind, VehicleConfiguration, VehicleBind(4), "");

	public:
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "Input TextureMesh");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet of the boundary");

	protected:
		void resetStates() override;
	};
}
