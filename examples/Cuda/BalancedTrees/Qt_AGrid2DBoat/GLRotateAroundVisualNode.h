/**
 * Copyright 2025 Lixin Ren
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
#include "Node.h"
#include "Topology/TextureMesh.h"
#include "RectangleRotateAroundAxis.h"

namespace dyno
{
	template<typename TDataType>
	class GLRotateAroundVisualNode : public Node
	{
		DECLARE_TCLASS(GLRotateAroundVisualNode, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		GLRotateAroundVisualNode();
		~GLRotateAroundVisualNode() override;

	public:
		void resetStates() override;
		void updateStates() override;
		bool validateInputs() override;

		DEF_NODE_PORTS(RectangleRotateAroundAxis<TDataType>, Shape, "");

		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "Input TextureMesh");
		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");
	};	
};
