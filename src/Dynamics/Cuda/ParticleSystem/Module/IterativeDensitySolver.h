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
#include "Algorithm/Arithmetic.h"

#include "Collision/Attribute.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/**
	 * @brief This is an implementation of the iterative density solver integrated into the Position Based Dynamics framework.
	 *
	 * For details, refer to "Position Based Fluids" by Macklin and M¡§uller, ACM TOG, 2013
	 *
	*/
	template<typename TDataType>
	class IterativeDensitySolver : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(IterativeDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		IterativeDensitySolver();
		~IterativeDensitySolver() override;

	public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input particle position");

		/**
		 * @brief Particle velocities
		 */
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");

		/**
		 * @brief Attribute
		 * Particle attribute
		 */
		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		/**
		 * @brief Neighboring particles' ids
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		/**
		 * @brief Final particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

	public:
		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR(Real, Kappa, Real(1), "");

	protected:
		void compute() override;

	public:
		void takeOneIteration();

		void updateVelocity();

	private:
		DArray<Real> mLamda;
		DArray<Coord> mDeltaPos;
		DArray<Coord> mPositionOld;


		Arithmetic<Real>* m_arithmetic;


	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(IterativeDensitySolver, TDataType)
}