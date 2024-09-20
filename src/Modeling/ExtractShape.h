/**
 * Copyright 2022 Yuzhong Guo
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
#include "Topology/TriangleSet.h"


namespace dyno
{
	/**
	 * @brief A class to merge TextureMeshs.
	 */

	template<typename TDataType>
	class ExtractShape: public Node
	{
		DECLARE_TCLASS(ExtractShape, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TopologyModule::Triangle Triangle;

		ExtractShape();

		~ExtractShape();

	public:

		DEF_VAR(std::vector<int>, ShapeId, std::vector<int>{0},"");

		DEF_VAR(std::vector<Transform3f>, ShapeTransform, std::vector<Transform3f>{Transform3f()}, "");

		DEF_VAR(bool, Offset, true, "");

		DEF_INSTANCE_IN(TextureMesh, InTextureMesh, "Input TextureMesh");

		DEF_INSTANCE_STATE(TextureMesh, Result, "Output TextureMesh");

	protected:
		void resetStates() override;


	private:


	};



	IMPLEMENT_TCLASS(ExtractShape, TDataType);
}