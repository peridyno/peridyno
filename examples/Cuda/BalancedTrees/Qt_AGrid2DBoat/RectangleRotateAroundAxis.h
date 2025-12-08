/**
 * Copyright 2025 Lixin Ren
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
#include "BasicShapes/RectangleModel2D.h"
#include "Topology/TextureMesh.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class RectangleRotateAroundAxis : public RectangleModel2D<TDataType>
	{
		DECLARE_TCLASS(RectangleRotateAroundAxis, TDataType);

	public:
		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord2D;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;

		RectangleRotateAroundAxis();

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

		DEF_VAR(Real, InitialAngle, 0.0f, "");
		DEF_VAR(Real, RotationRadius, 1.0f, "");		
		DEF_VAR(int, Frequency, 3000, "");

	protected:
		void resetStates() override;
		void updateStates() override;

		void updateCenterAndRotation(Real y_rotation);

	private:
		TAlignedBox3D<Real> m_alignedBox;
		Coord3D m_center;
	};

	IMPLEMENT_TCLASS(RectangleRotateAroundAxis, TDataType);
}
