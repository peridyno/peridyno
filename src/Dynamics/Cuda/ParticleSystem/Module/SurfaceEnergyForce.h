/**
 * Copyright 2017-2024 Xiaowei He
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
#include "ParticleApproximation.h"

namespace dyno {

	/**
	 * @brief An implementation of the energy-based surface model proposed by He et al.[2024].
	 * 		  Refer to "Robust Simulation of Sparsely Sampled Thin Features in SPH-Based Free Surface Flows",  ACM TOG 2014, for more details
	 */
	template<typename TDataType>
	class SurfaceEnergyForce : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(SurfaceEnergyForce, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceEnergyForce();
		~SurfaceEnergyForce() override;
		
		DEF_VAR(Real, Kappa, Real(1), "Surface tension coefficient");

		DEF_VAR(Real, RestDensity, Real(1000), "Rest density");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		/**
		 * @brief Neighboring particles
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

	public:
		void compute() override;

	private:
		DArray<Real> mFreeSurfaceEnergy;
	};

	IMPLEMENT_TCLASS(SurfaceEnergyForce, TDataType);
}