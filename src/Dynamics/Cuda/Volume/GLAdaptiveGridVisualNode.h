/**
 * Copyright 2022 Lixin Ren
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

#include "Topology/AdaptiveGridSet.h"
#include "Topology//TriangleSet.h"
//#include "Volume/VolumeOctree.h"

namespace dyno
{
	template<typename TDataType>
	class GLAdaptiveGridVisualNode : public Node
	{
		DECLARE_TCLASS(GLAdaptiveGridVisualNode, TDataType)
	public:
		DECLARE_ENUM(PointData,
			AGrid_Node = 0,
			AGrid_Vertex = 1
		);

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GLAdaptiveGridVisualNode();
		~GLAdaptiveGridVisualNode() override;

	public:
		bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		DEF_ENUM(PointData, Type, PointData::AGrid_Node, "");
		DEF_VAR(Real, PointSize, 0.1f, "The size of points");

		DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, AdaptiveVolume, "SDF Voxel Octree");
		DEF_ARRAY_IN(Real, AGridSDF, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(PointSet<DataType3f>, Leafs, "Leafs Voxel Octree");
		DEF_ARRAY_STATE(Real, LeafsValue, DeviceType::GPU, "The value of SDFOctree Leafs");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, LeafsEdge, "A set of triangles or edges");
	};
};
