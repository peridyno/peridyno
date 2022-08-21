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
#include <Node.h>

#include <Topology/TriangleSet.h>

namespace dyno
{
	template<typename TDataType>
	class GLSurfaceVisualNode : public Node
	{
		DECLARE_TCLASS(GLSurfaceVisualNode, TDataType)
	public:
		typedef typename TDataType::Coord Coord;

		GLSurfaceVisualNode();
		~GLSurfaceVisualNode() override;

	public:
		std::string caption() override;

		std::string getNodeType() override;

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "A set of triangles");
		DEF_VAR(Vec3f, Color, Vec3f(0.8f), "Color");
	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(GLSurfaceVisualNode, TDataType)
};
