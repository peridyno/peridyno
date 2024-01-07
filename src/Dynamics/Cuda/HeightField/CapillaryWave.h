/**
 * Copyright 2017-2022 hanxingyixue
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
#include "Topology/HeightField.h"

namespace dyno
{
	/**
	 * Simulation of a square capillary wave using the shallow water equation (SWE)
	 */
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		DECLARE_TCLASS(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		CapillaryWave();
		~CapillaryWave() override;

	public:
		DEF_VAR(Real, WaterLevel, 2, "");

		DEF_VAR(uint, Resolution, 512, "");

		DEF_VAR(Real, Length, 512.0f, "The simulated region size in meters");

	public:
		DEF_ARRAY2D_STATE(Coord4D, Height, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "Height field");

	public:
		void setOriginX(int x) { mOriginX = x; }
		void setOriginY(int y) { mOriginY = y; }

		int getOriginX() { return mOriginX; }
		int getOriginZ() { return mOriginY; }

		Real getRealGridSize() { return mRealGridSize; }

		Coord2D getOrigin() { return Coord2D(mOriginX * mRealGridSize, mOriginY * mRealGridSize); }

		//TODO: make improvements
		void moveDynamicRegion(int nx, int ny);

	protected:
		void resetStates() override;

		void updateStates() override;

	protected:
		DArray2D<Coord4D> mDeviceGrid;
		DArray2D<Coord4D> mDeviceGridNext;

	private:
		Real mRealGridSize;

		int mOriginX = 0;
		int mOriginY = 0;
	};

	IMPLEMENT_TCLASS(CapillaryWave, TDataType)
}