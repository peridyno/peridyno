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
#include "Node.h"
#include "Module/TopologyMapping.h"

#include "Topology/TextureMesh.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class TextureMeshToTriangleSet : public TopologyMapping
	{
		DECLARE_TCLASS(TextureMeshToTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::Transform<Real, 3> Transform;

		TextureMeshToTriangleSet();
		~TextureMeshToTriangleSet() override;

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

		DEF_ARRAYLIST_IN(Transform, Transform, DeviceType::GPU, "");
 
 		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		bool apply() override;
	};

	template<typename TDataType>
	class TextureMeshToTriangleSetNode : public Node
	{
		DECLARE_TCLASS(TextureMeshToTriangleSetNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TextureMeshToTriangleSetNode();

		std::string caption() override { return "TextureMeshToTriangleSet"; }

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		std::shared_ptr<TextureMeshToTriangleSet<TDataType>> mTM2TS;
	};
}