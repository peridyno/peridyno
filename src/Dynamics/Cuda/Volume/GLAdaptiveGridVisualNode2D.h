/**
 * Copyright 2024 Lixin Ren
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

#include "Topology/AdaptiveGridSet2D.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class GLAdaptiveGridVisualNode2D : public Node
	{
		DECLARE_TCLASS(GLAdaptiveGridVisualNode2D, TDataType)
	public:
		DECLARE_ENUM(EdgeType,
			Quadtree_Edge = 0,
			Quadtree_Neighbor = 1
			);

		DECLARE_ENUM(ProjectionPlane,
			XY_Plane = 0,
			XZ_Plane = 1,
			YZ_Plane = 2
		);

		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;

		GLAdaptiveGridVisualNode2D();
		~GLAdaptiveGridVisualNode2D() override;

	public:
		bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		DEF_ENUM(EdgeType, EType, EdgeType::Quadtree_Edge, "");
		DEF_ENUM(ProjectionPlane, PPlane, ProjectionPlane::XY_Plane, "");

		DEF_INSTANCE_IN(AdaptiveGridSet2D<TDataType>, AdaptiveVolume, "");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Grids, "A set of triangles or edges");
	};	
};
