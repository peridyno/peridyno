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
#include "Node/ParametricModel.h"

#include "Topology/QuadSet.h"

namespace dyno
{
	template<typename TDataType>
	class CubeModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(CubeModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CubeModel();

		std::string caption() override { return "Cube"; }

		NBoundingBox boundingBox() override;

	public:
		DEF_VAR(Coord, Length, Real(1), "Edge length");

		DEF_VAR(Vec3i, Segments, Vec3i(1, 1, 1), "");

		DEF_INSTANCE_STATE(QuadSet<TDataType>, QuadSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR_OUT(TOrientedBox3D<Real>, Cube,  "");

	protected:
		void resetStates() override;

	private:
		void varChanged();

	};

	IMPLEMENT_TCLASS(CubeModel, TDataType);
}