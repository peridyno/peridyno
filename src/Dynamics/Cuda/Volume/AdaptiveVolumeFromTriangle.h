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
#include "Volume/AdaptiveVolume.h"


namespace dyno 
{

	template<typename TDataType>
	class AdaptiveVolumeFromTriangle : public AdaptiveVolume<TDataType>
	{
		DECLARE_TCLASS(AdaptiveVolumeFromTriangle, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveVolumeFromTriangle() {};
		~AdaptiveVolumeFromTriangle() override {};

		void load(std::string filename);

		void resetStates() override;
		void updateStates() override;

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "The triangles of closed surface");
		DEF_VAR(int, AABBPadding, 0, "Padding of each triangle`s AABB");
		DEF_VAR(Coord, ForwardVector, Coord(0), "The distance and direction of topology move");
		DEF_VAR_STATE(bool, DynamicState, false, "");

	private:
		void initParameter();
		void computeSeeds();
	};
}