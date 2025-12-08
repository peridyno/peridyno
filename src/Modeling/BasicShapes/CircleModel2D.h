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
#include "BasicShape2D.h"

namespace dyno
{
	template<typename TDataType>
	class CircleModel2D : public BasicShape2D<TDataType>
	{
		DECLARE_TCLASS(CircleModel2D, TDataType);

	public:
		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord2D;
		typedef typename Vector<Real, 2> Coord2D;

		CircleModel2D();

		BasicShapeType2D getShapeType() override { return BasicShapeType2D::CIRCLE; }

		DEF_VAR(Real, Radius, 0.5, "Circle radius");
		DEF_VAR(Coord2D, Center2D, Real(1), "Circle center");

		DEF_VAR_OUT(TCircle2D<Real>, Circle, "");

	protected:
		void resetStates() override;

	};

	IMPLEMENT_TCLASS(CircleModel2D, TDataType);
}
