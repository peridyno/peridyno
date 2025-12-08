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
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class AdaptiveVolume : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveVolume();
		~AdaptiveVolume() override;

		std::string getNodeType() override { return "Adaptive Volumes"; }

	public:
		DEF_VAR(Real, Dx, 0.1, "Spacing of the Finest Grids");
		DEF_VAR(Level, MaxLevel, Level(0), "Max Level of region of interest");

		DEF_VAR_STATE(Coord, Origin, Coord(0.0, 0.0, 0.0), "Origin of region of interest");
		DEF_ARRAY_STATE(OcKey, pMorton, DeviceType::GPU, "The morton of increased seeds");

		DEF_INSTANCE_STATE(AdaptiveGridSet<TDataType>, AGridSet, "");
		DEF_ARRAY_STATE(Real, AGridSDF, DeviceType::GPU, "");

	private:

	};
}