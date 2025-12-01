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
	class OverscanGLAdaptiveXYPlane : public Node
	{
		DECLARE_TCLASS(OverscanGLAdaptiveXYPlane, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		OverscanGLAdaptiveXYPlane();
		~OverscanGLAdaptiveXYPlane() override;



	public:
		bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		DEF_VAR(Real, ZPos, 0.0f, "The position of z-axis");
		DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, AdaptiveVolume, "SDF Voxel Octree");

		//DEF_ARRAY_IN(Coord, ZPos, DeviceType::GPU, "");

		//DEF_INSTANCE_STATE(PointSet<TDataType>, LeafNodes, "Leafs Voxel Octree");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Grids, "A set of triangles or edges");

		DEF_VAR(uint, FirstFrame, 500, "");
		DEF_VAR(uint, MovingIntervalFrame, 1, "");
		DEF_VAR(Real, MovingStep, 0.01f, "");
		DEF_VAR(Real, LowerBound, -1.0f, "");
		DEF_VAR(Real, UpperBound, 1.0f, "");

	private:

		Real Interim(Real variable)
		{
			Real step = this->varMovingStep()->getValue();
			Real base = 0.15f;

			if (abs(variable) < 0.1f)
			{
				return 10 * step * abs(variable) + 0.5f * step;
			}
			else if (abs(1.0 - variable) < 0.1f)
			{
				return 10 * step * abs(1.0 - variable) + 0.5f * step;
			}
			else {
				return step;
			}

		};

		uint mFrameNumber = 0;

		bool mScanDirection = true;

		Real mTempPlane = 0.0f;
	};
};
