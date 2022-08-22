/**
 * Copyright 2021 Yue Chang
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
#include "RigidBodyShared.h"

namespace dyno
{
	/**
	 * @brief Implementation of an iterative constraint solver for rigid body dynamics with contact.
	 * 			Refer to "Iterative Dynamics with Temporal Coherence" by Erin Catto, 2005.
	 */
	template<typename TDataType>
	class IterativeConstraintSolver : public ConstraintModule
	{
		DECLARE_TCLASS(IterativeConstraintSolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		
		typedef typename Quat<Real> TQuat;
		typedef typename TContactPair<Real> ContactPair;
		typedef typename TConstraintPair<Real> Constraint;
		
		IterativeConstraintSolver();
		~IterativeConstraintSolver();

		void constrain() override;

	public:
		DEF_VAR(bool, FrictionEnabled, true, "");

		DEF_VAR(uint, IterationNumber, 10, "");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Velocity of rigid bodies");

		DEF_ARRAY_IN(Coord, AngularVelocity, DeviceType::GPU, "Angular velocity of rigid bodies");

		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_ARRAY_IN(Matrix, Inertia, DeviceType::GPU, "Interial matrix");

		DEF_ARRAY_IN(Matrix, InitialInertia, DeviceType::GPU, "Interial matrix");

		DEF_ARRAY_IN(TQuat, Quaternion, DeviceType::GPU, "Quaternion");

		DEF_ARRAY_IN(ContactPair, Contacts, DeviceType::GPU, "");

	private:
		void initializeJacobian(Real dt);

	private:
		DArray<Coord> mJ;		//Jacobian
		DArray<Coord> mB;		//B = M^{-1}J^T
		DArray<Coord> mAccel;

		DArray<Real> mEta;		//eta
		DArray<Real> mD;		//diagonal elements of JB
		DArray<Real> mLambda;	//contact impulse

		DArray<Real> mContactNumber;

		DArray<Matrix> mInitialInertia;

		DArray<Constraint> mAllConstraints;
	};
}