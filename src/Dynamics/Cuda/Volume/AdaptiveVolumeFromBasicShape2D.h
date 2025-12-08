/**
 * Copyright 2023 Lixin Ren
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

#include "AdaptiveVolume2D.h"
#include "Module/MSTsGenerator2D.h"
#include "Module/MSTsGeneratorLocalUpdate2D.h"
#include "BasicShapes/BasicShape2D.h"

namespace dyno {

	template<typename TDataType>
	class AdaptiveVolumeFromBasicShape2D : public AdaptiveVolume2D<TDataType>
	{
		DECLARE_TCLASS(AdaptiveVolumeFromBasicShape2D, TDataType)
	public:
		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;

		AdaptiveVolumeFromBasicShape2D();
		~AdaptiveVolumeFromBasicShape2D();

		DEF_VAR(Real, NarrowWidth, 0.1, "");

		DEF_VAR(bool, IsHollow, true, "");
		DEF_VAR(Coord2D, LowerBound, Real(REAL_MAX), "The upper bounds of the calculation domain");
		DEF_VAR(Coord2D, UpperBound, Real(REAL_MIN), "The lower bounds of the calculation domain");

		DEF_NODE_PORTS(BasicShape2D<TDataType>, Shape, "");

		DEF_ARRAY_IN(Coord3D, Particles, DeviceType::GPU, "");

		DEF_VAR(bool, DynamicMode, false, "");

	protected:
		void resetStates() override;
		void updateStates() override;
		bool validateInputs() override;

	private:
		void initParameter();
		void computeSeeds(DArray<uint>& marker);
		void computeSeedsFromScratch();
		void computeSeedsDynamicUpdate();
		void circleSeeds(DArray<uint>& marker, Coord2D center, Real radius);
		void rectangleSeeds(DArray<uint>& marker, TOrientedBox2D<Real>& obox);

		std::shared_ptr<MSTsGenerator2D<TDataType>> mMSTGen = nullptr;
		std::shared_ptr<MSTsGeneratorLocalUpdate2D<TDataType>> mMSTGenLocal = nullptr;

		Coord2D m_origin;
		Level m_levelmax;
	};

}
