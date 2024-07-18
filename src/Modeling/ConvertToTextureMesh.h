/**
 * Copyright 2022 Xiaowei He
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
#include "BasicShapes/BasicShape.h"

#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"
#include "Topology/TextureMesh.h"

#include "FilePath.h"


namespace dyno
{
	template<typename TDataType>
	class ConvertToTextureMesh : public Node
	{
		DECLARE_TCLASS(CubeModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		ConvertToTextureMesh();

	public:

		DEF_VAR(FilePath, DiffuseTexture, "", "");
		DEF_VAR(FilePath, NormalTexture, "", "");
		DEF_VAR(bool,UseBoundingTransform,false,"Active MovePointsToCenter,useBoundingTransform");
		DEF_INSTANCE_IN(TopologyModule, Topology, "");

		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");

	protected:
		void resetStates() override;

	private:


	};

	IMPLEMENT_TCLASS(ConvertToTextureMesh, TDataType);
}