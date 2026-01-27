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

#include "RigidBodySystem.h"

#include "Topology/TriangleSet.h"

namespace dyno 
{
	template<typename TDataType>
	class MultibodySystem : public RigidBodySystem<TDataType>
	{
		DECLARE_TCLASS(MultibodySystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename ::dyno::Pair<uint, uint> BindingPair;

		MultibodySystem();
		~MultibodySystem() override;

	public:
		/**
		 * @brief Creates multiple vehicles and specifies the transformations for each vehicle
		 */
		DEF_VAR(std::vector<Transform3f>, VehiclesTransform, std::vector<Transform3f>{Transform3f()}, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet of the boundary");

	public:
		DEF_NODE_PORTS(RigidBodySystem<TDataType>, Vehicle, "");

		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

		void postUpdateStates() override;

		bool validateInputs() override;
	};
}
