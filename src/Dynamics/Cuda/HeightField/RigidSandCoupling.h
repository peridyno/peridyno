/**
 * Copyright 2024 Xiaowei He
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

#include "HeightField/GranularMedia.h"

#include "RigidBody/RigidBodySystem.h"

namespace dyno
{
	/**
	 * @brief This class implements a coupling between a granular media and a rigid body system
	 */

	template<typename TDataType>
	class RigidSandCoupling : public Node
	{
		DECLARE_TCLASS(RigidWaterCoupling, TDataType)
	public:
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		RigidSandCoupling();
		~RigidSandCoupling() override;

	public:
		DEF_NODE_PORT(GranularMedia<TDataType>, GranularMedia, "Granular media");
		DEF_NODE_PORT(RigidBodySystem<TDataType>, RigidBodySystem, "Rigid body system");

	protected:
		void resetStates() override;
		void updateStates() override;
	};

	IMPLEMENT_TCLASS(RigidSandCoupling, TDataType)
}
