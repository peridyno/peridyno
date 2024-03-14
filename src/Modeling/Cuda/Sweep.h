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
#include "Ramp.h"

namespace dyno
{


	template<typename TDataType>
	class SweepModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(SweepModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SweepModel();

	public:

		DEF_VAR(Real, Radius, 0.6, "Cylinder radius");

		DEF_VAR(Ramp, CurveRamp,Ramp::Direction::x, "");

		DEF_VAR(bool, ReverseNormal, false, "ReverseNormal");

		DEF_VAR(bool, DisplayPoints, false, "DisplayPoints");

		DEF_VAR(bool, DisplayWireframe, false, "DisplayWireframe");

		DEF_VAR(bool, DisplaySurface, true, "DisplaySurface");
		
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, Spline, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, Curve, "");



		Vec3f RealScale();

	protected:
		void resetStates() override;

		void varChanged();

		void displayChanged();


		float currentIndex = 0;
		float totalIndex = 0;
	};



	IMPLEMENT_TCLASS(SweepModel, TDataType);
}