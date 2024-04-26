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
#include "BasicShape.h"

#include "Topology/PointSet.h"
#include "Topology/EdgeSet.h"

namespace dyno 
{
	/**
	 * This example demonstrates how to compute the contact manifold between two basic shapes
	 */
	template<typename TDataType>
	class CollisionDetector : public Node
	{
		DECLARE_TCLASS(CollisionDetector, TDataType);
	public:
		typedef typename TDataType::Coord Coord;

		CollisionDetector();
		~CollisionDetector() override {};

		std::string getNodeType() override { return "Collision"; }

	public:
		DEF_NODE_PORT(BasicShape<TDataType>, ShapeA, "");

		DEF_NODE_PORT(BasicShape<TDataType>, ShapeB, "");

	public:
		DEF_INSTANCE_STATE(PointSet<TDataType>, Contacts, "");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Normals, "");

	protected:
		void resetStates() override;

		bool validateInputs() override;
	};
}