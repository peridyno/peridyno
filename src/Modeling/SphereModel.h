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
 *
 * Revision history:
 *
 * 2024-02-03: replace TriangleSet with PolygonSet as the major state;
 */

#pragma once
#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"

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

		std::string caption() override { return "Sphere"; }

		NBoundingBox boundingBox() override;

	public:
		DEF_VAR(Coord, Center, 0, "Sphere center");

		DEF_VAR(Real, Radius, 0.5, "Sphere radius");

		DECLARE_ENUM(SphereType,
			Standard = 0,
			Icosahedron = 1);

		DEF_ENUM(SphereType, Type, SphereType::Standard, "Sphere type");

		DEF_VAR(uint, Latitude, 32, "Latitude");

		DEF_VAR(uint, Longitude, 32, "Longitude");

		DEF_INSTANCE_STATE(PolygonSet<TDataType>, PolygonSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TSphere3D<Real>, Sphere, "");

	protected:
		void resetStates() override;

	private:
		void varChanged();
	};



	IMPLEMENT_TCLASS(SphereModel, TDataType);
}