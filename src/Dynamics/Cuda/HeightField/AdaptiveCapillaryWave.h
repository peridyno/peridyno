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
#include "Node.h"
#include "HeightField/LandScape.h"
#include "Topology/AdaptiveGridSet2D.h"

namespace dyno {
	template<typename TDataType>
	class AdaptiveCapillaryWave : public Node
	{
		DECLARE_TCLASS(AdaptiveCapillaryWave, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		AdaptiveCapillaryWave();
		~AdaptiveCapillaryWave();

		DEF_INSTANCE_IN(AdaptiveGridSet2D<TDataType>, AGrid2D, "");

		DEF_VAR(Real, WaterLevel, 2, "");

		DEF_ARRAY_STATE(Coord4D, Heigh, DeviceType::GPU, "");

	protected:
		void resetStates() override;
		void updateStates() override;

		DArray<Coord4D> mDeviceGridNext;
	};
}