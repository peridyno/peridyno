/**
 * Copyright 2024 Xiaowe He
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

#include "LandScape.h"
#include "CapillaryWave.h"

namespace dyno
{
	/**
	 * Simulation of mountain torrents
	 */
	template<typename TDataType>
	class MountainTorrents : public CapillaryWave<TDataType>
	{
		DECLARE_TCLASS(MountainTorrents, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;
		typedef  Vector<Real, 4> Coord4D;

		MountainTorrents();
		~MountainTorrents() override;

		DEF_NODE_PORT(LandScape<TDataType>, Terrain, "");

		DEF_ARRAY2D_STATE(Real, InitialHeights, DeviceType::GPU, "Initial water heights");

	protected:
		void resetStates() override;

		void updateStates() override;

	private:
		DArray2D<Real> mInitialHeights;
	};

	IMPLEMENT_TCLASS(MountainTorrents, TDataType)
}