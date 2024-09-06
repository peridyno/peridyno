/**
 * Copyright 2022 Xiaowei He
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
#include "Node.h"

#include "Topology/SignedDistanceField.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class MarchingCubes : public Node
	{
		DECLARE_TCLASS(MarchingCubes, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MarchingCubes();

		~MarchingCubes() override;

	public:
		DEF_VAR(Real, IsoValue, Real(0), "Iso value");

		DEF_VAR(Real, GridSpacing, Real(0.05), "");

		DEF_INSTANCE_IN(SignedDistanceField<TDataType>, LevelSet, "A 3D signed distance field");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "An iso surface");

	protected:
		void resetStates() override;

		void updateStates() override;

	private:
		void constructSurfaceMesh();
	};

	IMPLEMENT_TCLASS(MarchingCubes, TDataType)
}