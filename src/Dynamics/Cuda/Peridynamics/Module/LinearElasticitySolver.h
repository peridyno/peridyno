/**
 * Copyright 2021 Xiaowei He
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
#include "Module/ConstraintModule.h"

#include "Peridynamics/Bond.h"

namespace dyno {

	/**
	  * @brief This is an implementation of elasticity based on projective peridynamics.
	  *		   For more details, please refer to[He et al. 2017] "Projective Peridynamics for Modeling Versatile Elastoplastic Materials"
	  */
	template<typename TDataType>
	class LinearElasticitySolver : public ConstraintModule
	{
		DECLARE_TCLASS(LinearElasticitySolver, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;

		LinearElasticitySolver();
		~LinearElasticitySolver() override;
		
		void constrain() override;

		virtual void solveElasticity();

	protected:
		void preprocess() override;

		/**
		 * @brief Correct the particle position with one iteration
		 * Be sure computeInverseK() is called as long as the rest shape is changed
		 */
		virtual void enforceElasticity();
		virtual void computeMaterialStiffness();

		void updateVelocity();
		void computeInverseK();

	public:
		/**
			* @brief Horizon
			* A positive number represents the radius of neighborhood for each point
			*/
		DEF_VAR_IN(Real, Horizon, "");

		DEF_VAR_IN(Real, TimeStep, "");

		/**
		 * @brief Reference position
		 */
		DEF_ARRAY_IN(Coord, X, DeviceType::GPU, "Rest Pos");

		/**
		 * @brief Deformed position
		 */
		DEF_ARRAY_IN(Coord, Y, DeviceType::GPU, "Particle position");

		/**
			* @brief Particle velocity
			*/
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Neighboring particles
		 * 
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		DEF_ARRAYLIST_IN(Bond, Bonds, DeviceType::GPU, "Reference shape");

	public:
		/**
		 * @brief Lame parameters
		 * m_lambda controls the isotropic part while mu controls the deviatoric part.
		 */
		DEF_VAR(Real, Mu, 0.001, "Lame parameters: mu");

		DEF_VAR(Real, Lambda, 0.01, "Lame parameters: lambda");

		DEF_VAR(uint, IterationNumber, 30, "Iteration number");

	protected:
		DArray<Real> mBulkStiffness;
		DArray<Real> mWeights;

		DArray<Coord> mDisplacement;
		DArray<Coord> mPosBuf;

		DArray<Matrix> mF;
		DArray<Matrix> mInvK;
	};
}