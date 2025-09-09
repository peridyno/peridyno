/**
 * Copyright 2017-2021 Xiaowei He
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
#include "Volume.h"
#include "Array/Array3D.h"

#include "Topology/TriangleSet.h"

namespace dyno {
	template<typename TDataType>
	class VolumeGenerator : public Volume<TDataType>
	{
		DECLARE_TCLASS(VolumeGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		VolumeGenerator();
		~VolumeGenerator() override;

		DEF_VAR(Real, Spacing, 0.05f, "");

		DEF_VAR(uint, Padding, 10, "");

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");
		//DEF_INSTANCE_IN(LevelSet<TDataType>, LevelSet_exact0, "");

		DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");

	protected:
		void resetStates() override;
		void updateStates() override;
	};
}
