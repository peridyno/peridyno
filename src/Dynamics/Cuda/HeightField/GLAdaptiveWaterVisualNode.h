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

#include "AdaptiveCapillaryWave.h"
#include "Topology/PolygonSet.h"

namespace dyno
{
	template<typename TDataType>
	class GLAdaptiveWaterVisualNode : public Node
	{
		DECLARE_TCLASS(GLAdaptiveWaterVisualNode, TDataType)
	public:
		DECLARE_ENUM(EdgeType,
			Quadtree_Edge = 0,
			Quadtree_Neighbor = 1
			);

		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		GLAdaptiveWaterVisualNode();
		~GLAdaptiveWaterVisualNode() override;

	public:
		bool validateInputs() override;

		void resetStates() override;
		void updateStates() override;

		DEF_INSTANCE_IN(AdaptiveGridSet2D<TDataType>, AGridSet, "");
		DEF_ARRAY_IN(Coord4D, Grid, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(PolygonSet<TDataType>, PolygonSet, "");

		DEF_ARRAY_STATE(OcKey, SeedMorton, DeviceType::GPU, "The morton of increased seeds");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Seeds, "");

		DEF_VAR(Real, WaterOffset, 0.0f, "");

	private:
		void generateWaterSurface();
	};	
};
