/**
 * Copyright 2025 Lixin Ren
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
#include "AdaptiveVolumeFromTriangle.h"
#include "Module/VolumeMacros.h"

namespace dyno 
{

	template<typename TDataType>
	class AdaptiveVolumeFromTriangleSDF : public AdaptiveVolumeFromTriangle<TDataType>
	{
		DECLARE_TCLASS(AdaptiveVolumeFromTriangleSDF, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveVolumeFromTriangleSDF();
		~AdaptiveVolumeFromTriangleSDF();

		void resetStates() override;
		void updateStates() override;

	private:
		void initialSDF();

		void computeSDF();

		DArray<GridType> m_GridType;
		DArray<AdaptiveGridNode> m_nodes;
		DArray<int> m_vertex_neighbor, m_node2ver;

	};
}
