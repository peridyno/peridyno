/**
 * Copyright 2024 Yuzhong Guo
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
#include "BasicShape.h"

#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"

namespace dyno
{
	template<typename TDataType>
	class CapsuleModel : public BasicShape<TDataType>
	{
		DECLARE_TCLASS(CapsuleModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CapsuleModel();

		std::string caption() override { return "Capsule"; }
		
		BasicShapeType getShapeType() override { return BasicShapeType::CAPSULE; }

		NBoundingBox boundingBox() override;

	public:
		DEF_VAR(Coord, Center, 0, "Sphere center");

		DEF_VAR(Real, Radius, 0.25, "Sphere radius");

		DEF_VAR(uint, Latitude, 20, "Latitude");

		DEF_VAR(uint, Longitude, 20, "Longitude");

		DEF_VAR(Real, Height, 0.5, "Height");

		DEF_VAR(uint, HeightSegment, 8, "HeightSegment");

		DEF_INSTANCE_STATE(PolygonSet<TDataType>, PolygonSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TCapsule3D<Real>, Capsule, "");

	protected:
		void resetStates() override;

	private:
		void varChanged();
	};



	IMPLEMENT_TCLASS(CapsuleModel, TDataType);
}