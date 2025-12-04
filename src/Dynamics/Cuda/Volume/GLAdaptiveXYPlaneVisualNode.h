/**
 * Copyright 2023 Lixin Ren
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
	class GLAdaptiveXYPlaneVisualNode : public Node
	{
		DECLARE_TCLASS(GLAdaptiveXYPlaneVisualNode, TDataType)
	public:
		DECLARE_ENUM(EdgeData,
			Octree_Edge = 0,
			Octree_Neighbor = 1
			);

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GLAdaptiveXYPlaneVisualNode();
		~GLAdaptiveXYPlaneVisualNode() override;

	public:
		bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		DEF_VAR(Real, ZPos, 0.0f, "The position of z-axis");
		DEF_ENUM(EdgeData, Type, EdgeData::Octree_Edge, "");

		DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, AdaptiveVolume, "SDF Voxel Octree");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Grids, "A set of triangles or edges");


		//DEF_ARRAY_STATE(Coord, LeafPosition, DeviceType::GPU, "Leafs position");
		//DEF_ARRAY_STATE(Coord, LeafScale, DeviceType::GPU, "Leafs scale");
		//DEF_ARRAY_STATE(Real, LeafsValue, DeviceType::GPU, "The value of SDFOctree Leafs");

	};	
};
