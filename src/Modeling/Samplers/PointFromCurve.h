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
	class PointFromCurve : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(PointFromCurve, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointFromCurve();

	public:

		DEF_VAR(Real, UniformScale, 1, "Uniform Scale");

		DEF_VAR(Curve, Curve, Curve::CurveMode::Open, "");

		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "Points");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "Curve");



	protected:
		void resetStates() override;


	private:
		void varChanged();

	};



	IMPLEMENT_TCLASS(PointFromCurve, TDataType);
}