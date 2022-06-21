/**
 * Copyright 2022 Xiaowei He
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
#include "ParametricModel.h"

namespace dyno
{
	template<typename TDataType>
	class SphereModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(SphereModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SphereModel();

	public:
		DEF_VAR(Coord, Center, 0, "Sphere center");
		DEF_VAR(Real, Radius, 0, "Sphere radius");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TSphere3D<Real>, Sphere, "");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(SphereModel, TDataType);
}