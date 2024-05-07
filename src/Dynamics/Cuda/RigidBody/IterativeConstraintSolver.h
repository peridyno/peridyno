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
#include "Topology/Joint.h"

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
		
		typedef typename ::dyno::Quat<Real> TQuat;
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename ::dyno::TConstraintPair<Real> Constraint;

		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		typedef typename HingeJoint<Real> HingeJoint;
		typedef typename FixedJoint<Real> FixedJoint;
		typedef typename PointJoint<Real> PointJoint;
	
		IterativeConstraintSolver();
		~IterativeConstraintSolver();

	public:
		DEF_VAR(bool, FrictionEnabled, true, "");

		DEF_VAR(bool, GravityEnabled, true, "");

		DEF_VAR(Real, GravityValue, 9.8, "");

		DEF_VAR(Real, FrictionCoefficient, 100, "");

		DEF_VAR(Real, Slop, 0.0001, "");

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

		DEF_ARRAY_IN(BallAndSocketJoint, BallAndSocketJoints, DeviceType::GPU, "Ball And Socket Joints");

		DEF_ARRAY_IN(SliderJoint, SliderJoints, DeviceType::GPU, "Slider Joints");

		DEF_ARRAY_IN(HingeJoint, HingeJoints, DeviceType::GPU, "Hinge Joints");

		DEF_ARRAY_IN(FixedJoint, FixedJoints, DeviceType::GPU, "Fixed Joints");

		DEF_ARRAY_IN(PointJoint, PointJoints, DeviceType::GPU, "Point Joints");

	protected:
		void constrain() override;

	private:
		void initializeJacobian(Real dt);

	private:
		DArray<Coord> mJ;		//Jacobian
		DArray<Coord> mB;		//B = M^{-1}J^T

		DArray<Coord> mImpulseC;
		DArray<Coord> mImpulseExt;

		DArray<Real> mEta;		//eta
		DArray<Real> mD;		//diagonal elements of JB
		DArray<Real> mLambda;	//contact impulse

		DArray<Real> mLambda_old;

		DArray<Real> mDiff;
		CArray<Real> mDiffHost;

		DArray<ContactPair> mContactsInLocalFrame;

		DArray<Constraint> mAllConstraints;

		DArray<int> mContactNumber;
		DArray<int> mJointNumber;


	};
}