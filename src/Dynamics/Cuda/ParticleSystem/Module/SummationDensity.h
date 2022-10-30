/**
 * Copyright 2017-2021 Xiaowei He
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
	 * @brief The standard summation density
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class SummationDensity : public virtual ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(SummationDensity, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SummationDensity();
		~SummationDensity() override {};

		void compute() override;
	
	public:
		void compute(
			DArray<Real>& rho,
			DArray<Coord>& pos,
			DArrayList<int>& neighbors,
			Real smoothingLength,
			Real mass);

		void compute(
			DArray<Real>& rho,
			DArray<Coord>& pos,
			DArray<Coord>& posQueried,
			DArrayList<int>& neighbors,
			Real smoothingLength,
			Real mass);

	public:
		DEF_VAR(Real, RestDensity, 1000, "Rest Density");

		///Define inputs
		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Other, DeviceType::GPU, "Particle position");

		/**
		 * @brief Neighboring particles
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		///Define outputs
		/**
		 * @brief Particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Return the particle density");

		Real getParticleMass() {
			return m_particle_mass;
		}
		
	private:
		void calculateParticleMass();

		Real m_particle_mass;
		Real m_factor;
	};

	IMPLEMENT_TCLASS(SummationDensity, TDataType)
}