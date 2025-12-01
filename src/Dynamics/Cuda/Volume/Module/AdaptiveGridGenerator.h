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
#include "Module/ComputeModule.h"
#include "Topology/AdaptiveGridSet.h"
#include "DeclareEnum.h"

namespace dyno
{

	template<typename TDataType>
	class AdaptiveGridGenerator : public ComputeModule
	{
		//DECLARE_TCLASS(AdaptiveGridGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DECLARE_ENUM(OCTREETYPE,
		VERTEX_BALANCED = 0,
			EDGE_BALANCED = 1,
			FACE_BALANCED = 2,
			NON_BALANCED = 3,
			STRONG_BALANCED = 4);

		DECLARE_ENUM(NeighborMode,
			SIX_NEIGHBOR = 0,
			TEWNTY_SEVEN_NEIGHBOR = 1);

		AdaptiveGridGenerator();
		~AdaptiveGridGenerator() override;

		//DEF_VAR_IN(uint, FrameNumber, "Frame number");

		DEF_VAR(Level, LevelNum, (Level)4, "The Level Number of region of interest");
		DEF_ENUM(OCTREETYPE, OctreeType, OCTREETYPE::FACE_BALANCED, "octree type");
		DEF_ENUM(NeighborMode, NeighMode, NeighborMode::SIX_NEIGHBOR, "Neighbor Mode");

		DEF_ARRAY_IN(OcKey, pMorton, DeviceType::GPU, "The morton of increased seeds");
		DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, AGridSet, "");

		void compute() override;
	};
}
