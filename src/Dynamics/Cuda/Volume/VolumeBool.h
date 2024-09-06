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

namespace dyno {
	template<typename TDataType>
	class VolumeBool : public Node
	{
		DECLARE_TCLASS(VolumeBool, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeBool();
		~VolumeBool() override;

		DECLARE_ENUM(BoolType,
			Intersect = 0,
			Union = 1,
			Minus = 2,
			);

	protected:
		void resetStates() override;

		void CalcuSDFGrid(DistanceField3D<TDataType> aDistance,
			DistanceField3D<TDataType> bDistance,
			DistanceField3D<TDataType>& tDistance);

	public:
		DEF_INSTANCE_IN(SignedDistanceField<TDataType>, A, "");
		DEF_INSTANCE_IN(SignedDistanceField<TDataType>, B, "");

		DEF_INSTANCE_OUT(SignedDistanceField<TDataType>, SDF, "");

		DEF_VAR_IN(Real, Spacing, "");
		DEF_VAR_IN(uint, Padding, "");

		DEF_ENUM(BoolType, BoolType, BoolType::Union, "Volume Bool Type");
	};
}
