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

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{


	template<typename TDataType>
	class ConeModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(ConeModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ConeModel();

	public:
		DEF_VAR(unsigned, Columns, 24, "Cone Columns");

		DEF_VAR(unsigned, Row, 4, "Cone Row");

		DEF_VAR(Real, Radius, 0.6, "Cone radius");

		DEF_VAR(Real, Height, 0.9, "Cone Height");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TCone3D<Real>, Cone, "");

	protected:
		void resetStates() override;
	};



	IMPLEMENT_TCLASS(ConeModel, TDataType);
}