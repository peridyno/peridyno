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
#include "Module/ConstraintModule.h"

namespace dyno 
{
	/**
	 * @brief A linear damping model
	 *
	 * @tparam TDataType
	 */
	template<typename TDataType>
	class LinearDamping : public ConstraintModule
	{
		DECLARE_TCLASS(LinearDamping, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		LinearDamping();
		~LinearDamping() override;

		void constrain() override;

	public:
		DEF_VAR(Real, DampingCoefficient, 0.9, "");

		/**
		* @brief Particle velocity
		*/
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
	};

	IMPLEMENT_TCLASS(LinearDamping, TDataType)
}
