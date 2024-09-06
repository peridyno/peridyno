/**
 * Copyright 2024 Xiaowei He
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
#include "CapillaryWave.h"
#include "Vessel.h"

namespace dyno
{
	/**
	 * Simulation of the region of recirculating flow immediately behind a moving or stationary vessel.
	 */
	template<typename TDataType>
	class Wake : public CapillaryWave<TDataType>
	{
		DECLARE_TCLASS(Wake, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		Wake();
		~Wake() override;

	public:
		DEF_VAR(Real, Magnitude, 0.2, "A parameter to control the strength of the tails");

		DEF_NODE_PORT(Vessel<TDataType>, Vessel, "");

	protected:
		void updateStates() override;

	private:
		void addTrails();
		
		DArray2D<Real> mWeight;
		DArray2D<Coord2D> mSource;
	};
}