/**
 * Copyright 2022 Shusen Liu
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

#include "Topology/TriangleSet.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	class SphereModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(SphereModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;


		DECLARE_ENUM(SphereMode,
			Theta = 0,
			RowAndColumns = 1);


		SphereModel();

	public:
		DEF_VAR(Coord, Center, 0, "Sphere center");

		DEF_VAR(Real, Radius, 1, "Sphere radius");

		//DEF_VAR(Real, triangleLength, 0.5, "Length of triangle edge");

		DEF_ENUM(SphereMode, SphereMode, SphereMode::RowAndColumns, "ScaleMode");

		DEF_VAR(Real, Theta, 0.3, "Angle");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TSphere3D<Real>, Sphere, "");

		DEF_VAR(unsigned, Columns, 50, "Sphere Columns");

		DEF_VAR(unsigned, Row, 50, "Sphere Row");

		void disableRender();

		NBoundingBox boundingBox() override;

	protected:
		void resetStates() override;

	private:
		void varChanged();

		std::shared_ptr <GLSurfaceVisualModule> glModule;
	};



	IMPLEMENT_TCLASS(SphereModel, TDataType);
}