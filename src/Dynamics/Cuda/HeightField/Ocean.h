/**
 * Copyright 2017-2022 Xiaowei He
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
#include "OceanPatch.h"
#include "CapillaryWave.h"

namespace dyno
{
	template<typename TDataType>
	class Ocean : public Node
	{
		DECLARE_TCLASS(Ocean, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Ocean();
		~Ocean();

	public:
		DEF_VAR(uint, ExtentX, 1, "");
		DEF_VAR(uint, ExtentZ, 1, "");

		DEF_VAR(Real, WaterLevel, 0, "");

		DEF_NODE_PORT(OceanPatch<TDataType>, OceanPatch, "Ocean Patch");
		DEF_NODE_PORTS(CapillaryWave<TDataType>, CapillaryWave, "Capillary Wave");

		DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "Topology");

	protected:
		void resetStates() override;
		void updateStates() override;

		bool validateInputs() override;
	};

	IMPLEMENT_TCLASS(Ocean, TDataType)
}