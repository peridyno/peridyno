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

#include "Topology/TriangleSet.h"
#include "Topology/QuadSet.h"

namespace dyno
{
	template<typename TDataType>
	class PlaneModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(PlaneModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PlaneModel();

		std::string caption() override { return "Plane"; }

		NBoundingBox boundingBox() override;

	public:
		DEF_VAR(Real, LengthX, Real(1), "length X");
		DEF_VAR(Real, LengthZ, Real(1), "length Z");

		DEF_VAR(unsigned, SegmentX, unsigned(1), "Segment X");
		DEF_VAR(unsigned, SegmentZ, unsigned(1), "Segment Z");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(QuadSet<TDataType>, QuadSet, "");

	protected:
		void resetStates() override;

	private:
		void varChanged();

	};

	IMPLEMENT_TCLASS(PlaneModel, TDataType);
}