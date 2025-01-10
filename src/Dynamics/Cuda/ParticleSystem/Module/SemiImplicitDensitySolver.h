/**
 * Copyright 2023 Xiaowei He
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

#include "Collision/Attribute.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*!
	*	\class	SemiImplicitDensitySolver
	*	\brief	This class implements a semi-implicit successive substitution method to solve incompressibility.
	*/
	template<typename TDataType>
	class SemiImplicitDensitySolver : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(IterativeDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SemiImplicitDensitySolver();
		~SemiImplicitDensitySolver() override;

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
		void updatePosition();

		void updateVelocity();

	private:
		DArray<Coord> mPosBuf;
		DArray<Coord> mPosOld;

		DArray<Real> mDiagnals;

	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(SemiImplicitDensitySolver, TDataType)
}