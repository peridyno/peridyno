/**
 * Copyright 2017-2021 hanxingyixue
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
#include "Module/VolumeMacros.h"

namespace dyno {
	template<typename TDataType>
	class VolumeBoolean : public Node
	{
		DECLARE_TCLASS(VolumeBoolean, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeBoolean();
		~VolumeBoolean() override;

		std::string getNodeType() override;

		DEF_VAR(Real, Spacing, 0.01, "");
		DEF_VAR(uint, Padding, 5, "");

		DEF_ENUM(BoolType, BoolType, BoolType::Union, "Volume Bool Type");

	public:
		DEF_INSTANCE_IN(LevelSet<TDataType>, A, "");
		DEF_INSTANCE_IN(LevelSet<TDataType>, B, "");

		DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");

	protected:
		void resetStates() override;
	};
}
