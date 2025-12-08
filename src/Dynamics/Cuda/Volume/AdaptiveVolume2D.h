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
	class PositionNode
	{
	public:

		DYN_FUNC PositionNode()
		{
			surface_index = EMPTY;
			position_index = 0;
		}
		DYN_FUNC PositionNode(int surf, OcKey pos)
		{
			surface_index = surf;
			position_index = pos;
		}
		DYN_FUNC bool operator> (const PositionNode& ug) const
		{
			return position_index > ug.position_index;
		}
		DYN_FUNC bool isEmpty() { return surface_index == EMPTY; }

		int surface_index;
		OcKey position_index;
	};
	struct PositionCmp
	{
		DYN_FUNC bool operator()(const PositionNode& A, const PositionNode& B)
		{
			return A > B;
		}
	};

	template<typename TDataType>
	class AdaptiveVolume2D : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveVolume2D();
		~AdaptiveVolume2D() override;

	public:
		DEF_VAR(Real, Dx, 1.0, "Spacing of the Finest Grids");
		DEF_VAR(Level, MaxLevel, Level(0), "Max Level of region of interest");
		DEF_VAR(Level, LevelNum, Level(0), "Level Num of region of interest");

		DEF_ARRAY_STATE(OcKey, IncreaseMorton, DeviceType::GPU, "The morton of increased seeds");
		DEF_ARRAY_STATE(OcKey, DecreaseMorton, DeviceType::GPU, "The morton of decreased seeds");

		DEF_INSTANCE_STATE(AdaptiveGridSet2D<TDataType>, AGridSet, "");
		DEF_ARRAY_STATE(Real, AGridSDF, DeviceType::GPU, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:

	};
}