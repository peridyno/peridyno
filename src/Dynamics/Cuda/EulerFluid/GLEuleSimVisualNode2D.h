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
#include "EulerFluid/EulerianSim.h"

namespace dyno
{
	template<typename TDataType>
	class GLEuleSimVisualNode2D : public Node
	{
		DECLARE_TCLASS(GLEuleSimVisualNode2D, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GLEuleSimVisualNode2D();
		~GLEuleSimVisualNode2D() override;

	public:
		//bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		//DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, AdaptiveVolume, "The adaptive volume data of model");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "The velocity of SDFOctree Leafs");

		DEF_INSTANCE_IN(PointSet<DataType3f>, Leafs, "Leafs Voxel Octree");
		DEF_ARRAY_IN(Real, LeafsValue, DeviceType::GPU, "The value of SDFOctree Leafs");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "A set of edges for velocity");
	};	
};
