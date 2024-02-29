/**
 * Copyright 2022 Yuzhong Guo
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
#include "Node/ParametricModel.h"
#include "Topology/TriangleSet.h"

#include "FilePath.h"


namespace dyno
{


	template<typename TDataType>
	class GltfLoader : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(GltfLoader, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef unsigned char byte;


		GltfLoader();

	public:

		DEF_VAR(FilePath, FileName, "", "");
		DEF_VAR(unsigned,Test,8,"");

		//DefaultChannel
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Position");
		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "Normal");
		DEF_ARRAY_STATE(Coord, UV, DeviceType::GPU, "UV");
		
		//CustomChannel
		DEF_VAR(std::string, RealName_1, "", "RealName_1");
		DEF_VAR(std::string, RealName_2, "", "RealName_2");
		DEF_VAR(std::string, CoordName_1, "", "CoordName_1");
		DEF_VAR(std::string, CoordName_2, "", "CoordName_2");

		DEF_ARRAY_STATE(Real, RealChannel_1, DeviceType::GPU, "RealChannel_1");
		DEF_ARRAY_STATE(Real, RealChannel_2, DeviceType::GPU, "RealChannel_2");
		DEF_ARRAY_STATE(Coord, CoordChannel_1, DeviceType::GPU, "CoordChannel_1");
		DEF_ARRAY_STATE(Coord, CoordChannel_2, DeviceType::GPU, "CoordChannel_1");

		//
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override 
		{
			varChanged();
		}

	private:
		void varChanged();

	
	};



	IMPLEMENT_TCLASS(GltfLoader, TDataType);
}