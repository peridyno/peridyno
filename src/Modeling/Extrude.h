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
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "Curve.h"

namespace dyno
{
	template<typename TDataType>
	class ExtrudeModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(ExtrudeModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ExtrudeModel();

		void varChanged();

		void displayChanged();

	public:

		DEF_VAR(unsigned, Row, 4, "Row");

		DEF_VAR(Real, Height, 1, "Height");

		DEF_VAR(bool, ReverseNormal, false, "ReverseNormal");

		DEF_VAR(Curve, Curve, Curve::CurveMode::Close, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");



	protected:
		void resetStates() override;


	};

	IMPLEMENT_TCLASS(ExtrudeModel, TDataType);
}

