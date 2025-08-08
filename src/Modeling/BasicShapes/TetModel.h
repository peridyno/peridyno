/**
 * Copyright 2025 Xukun Luo
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
#include "BasicShape.h"

#include "Topology/TetrahedronSet.h"

namespace dyno
{
	// Right Angle Tetrahedron
	template<typename TDataType>
	class TetModel : public BasicShape<TDataType>
	{
		DECLARE_TCLASS(TetModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TetModel();

		std::string caption() override { return "Tet"; }

		BasicShapeType getShapeType() override { return BasicShapeType::TET; }

		NBoundingBox boundingBox() override;
	
	public:
		DEF_VAR(Coord, V0, Coord(0, 0, 0), "First vertex");
		DEF_VAR(Coord, V1, Coord(1, 0, 0), "Second vertex");
		DEF_VAR(Coord, V2, Coord(0, 1, 0), "Third vertex");
		DEF_VAR(Coord, V3, Coord(0, 0, 1), "Fourth vertex");

		DEF_INSTANCE_STATE(TetrahedronSet<TDataType>, TetSet, "");

		DEF_VAR_OUT(TTet3D<Real>, Tet,  "");

	protected: 
		void resetStates() override;

	private:
		void varChanged();

	};

	IMPLEMENT_TCLASS(TetModel, TDataType);
}