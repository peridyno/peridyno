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
//#include "VolumeOctreeGenerator.h"
#include "Module/ComputeModule.h"
#include "Topology/AdaptiveGridSet.h"
#include "Volume/Module/MSTsGenerator.h"

namespace dyno 
{
	//template<typename TDataType> class VolumeOctreeGenerator;
	template<typename TDataType> class AdaptiveGridSet;
	template<typename TDataType> class MSTsGenerator;

	template<typename TDataType>
	class NeighborPointQueryAdaptive : public ComputeModule
	{
		DECLARE_TCLASS(NeighborPointQueryAdaptive, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborPointQueryAdaptive();
		~NeighborPointQueryAdaptive() override;
		
		void compute() override;

	private:
		void requestDynamicNeighborIds();
		void initParameter();
		void construct();
		void initialNode();
		//void requestFixedSizeNeighborIds();

	public:
		DEF_VAR(uint, SizeLimit, 0, "Maximum number of neighbors");

		DEF_VAR(uint, SizeMin, 1, "Minimum number of particles per grid");

		//the radius of the finest grids
		DEF_VAR_IN(Real, Radius, "Search radius");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points to be required from");

		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

		DEF_INSTANCE_OUT(AdaptiveGridSet<TDataType>, AdaptiveGrids, "");

	private:
		//Level m_level;
		//Coord m_origin;
		DArray<OcKey> m_morton;

		DArray<Coord> m_node;
		DArrayList<int> m_neighbor;

		DArray<int> m_pIndex;

		DArray<int> m_index;
		DArray<int> m_counter;
		DArray<int> m_ids;

		std::shared_ptr<AdaptiveGridSet<TDataType>> mAGrid;

		std::shared_ptr<MSTsGenerator<TDataType>> mAGridGen;
	};
}