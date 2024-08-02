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
#include "Node.h"
#include "RigidBodyShared.h"

#include "Collision/Attribute.h"

#include <vector>
#include <iostream>
namespace dyno
{
	/*!
	*	\class	RigidBodySystem
	*	\brief	Implementation of a rigid body system containing a variety of rigid bodies with different shapes.
	*
	*/
	template<typename TDataType>
	class RigidBodySystem : virtual public Node
	{
		DECLARE_TCLASS(RigidBodySystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename dyno::TSphere3D<Real> Sphere3D;
		typedef typename dyno::TOrientedBox3D<Real> Box3D;
		typedef typename dyno::Quat<Real> TQuat;

		typedef typename dyno::TContactPair<Real> ContactPair;
		
		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		typedef typename HingeJoint<Real> HingeJoint;
		typedef typename FixedJoint<Real> FixedJoint;
		typedef typename PointJoint<Real> PointJoint;


		RigidBodySystem();
		~RigidBodySystem() override;

		std::shared_ptr<PdActor> addBox(
			const BoxInfo& box, 
			const RigidBodyInfo& bodyDef,
			const Real density = Real(100));

		std::shared_ptr<PdActor> addSphere(
			const SphereInfo& sphere,
			const RigidBodyInfo& bodyDef, 
			const Real density = Real(100));

		std::shared_ptr<PdActor> addTet(
			const TetInfo& tet,
			const RigidBodyInfo& bodyDef,
			const Real density = Real(100));

		std::shared_ptr<PdActor> addCapsule(
			const CapsuleInfo& capsule,
			const RigidBodyInfo& bodyDef,
			const Real density = Real(100));



		BallAndSocketJoint& createBallAndSocketJoint(
			std::shared_ptr<PdActor> actor1,
			std::shared_ptr<PdActor> actor2)
		{
			BallAndSocketJoint joint(actor1.get(), actor2.get());
			mHostJointsBallAndSocket.push_back(joint);

			return mHostJointsBallAndSocket[mHostJointsBallAndSocket.size() - 1];
		}

		SliderJoint& createSliderJoint(
			std::shared_ptr<PdActor> actor1,
			std::shared_ptr<PdActor> actor2)
		{
			SliderJoint joint(actor1.get(), actor2.get());
			mHostJointsSlider.push_back(joint);

			return mHostJointsSlider[mHostJointsSlider.size() - 1];
		}

		HingeJoint& createHingeJoint(
			std::shared_ptr<PdActor> actor1,
			std::shared_ptr<PdActor> actor2)
		{
			HingeJoint joint(actor1.get(), actor2.get());
			mHostJointsHinge.push_back(joint);

			return mHostJointsHinge[mHostJointsHinge.size() - 1];
		}

		FixedJoint& createFixedJoint(
			std::shared_ptr<PdActor> actor1,
			std::shared_ptr<PdActor> actor2)
		{
			FixedJoint joint(actor1.get(), actor2.get());
			mHostJointsFixed.push_back(joint);

			return mHostJointsFixed[mHostJointsFixed.size() - 1];
		}

		FixedJoint& createUnilateralFixedJoint(
			std::shared_ptr<PdActor> actor1)
		{
			FixedJoint joint(actor1.get());
			mHostJointsFixed.push_back(joint);

			return mHostJointsFixed[mHostJointsFixed.size() - 1];
		}

		PointJoint& createPointJoint(
			std::shared_ptr<PdActor> actor1)
		{
			PointJoint joint(actor1.get());
			mHostJointsPoint.push_back(joint);

			return mHostJointsPoint[mHostJointsPoint.size() - 1];
		}

		Mat3f pointInertia(Coord v1);

	protected:
		void resetStates() override;

		void updateTopology() override;

		void clearRigidBodySystem();

	public:
		DEF_VAR(bool, FrictionEnabled, true, "A toggle to control the friction");

		DEF_VAR(bool, GravityEnabled, true, "A toggle to control the gravity");

		DEF_VAR(Real, GravityValue, 9.8, "");

		DEF_VAR(Real, FrictionCoefficient, 200, "");

		DEF_VAR(Real, Slop, 0.0001, "");

		DEF_INSTANCE_STATE(DiscreteElements<TDataType>, Topology, "Topology");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

		/**
		 * @brief The initial offset of barycenters
		 */
		DEF_ARRAY_STATE(Coord, Offset, DeviceType::GPU, "Offset of barycenters");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Velocity of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, AngularVelocity, DeviceType::GPU, "Angular velocity of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_ARRAY_STATE(Matrix, Inertia, DeviceType::GPU, "Inertia matrix");

		DEF_ARRAY_STATE(TQuat, Quaternion, DeviceType::GPU, "Quaternion");

		DEF_ARRAY_STATE(CollisionMask, CollisionMask, DeviceType::GPU, "Collision mask for each rigid body");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "Rigid body attributes");

		DEF_ARRAY_STATE(Matrix, InitialInertia, DeviceType::GPU, "Initial inertia matrix");

	private:
		std::vector<RigidBodyInfo> mHostRigidBodyStates;

		std::vector<SphereInfo> mHostSpheres;
		std::vector<BoxInfo> mHostBoxes;
		std::vector<TetInfo> mHostTets;
		std::vector<CapsuleInfo> mHostCapsules;

		DArray<RigidBodyInfo> mDeviceRigidBodyStates;

		DArray<SphereInfo> mDeviceSpheres;
		DArray<BoxInfo> mDeviceBoxes;
		DArray<TetInfo> mDeviceTets;
		DArray<CapsuleInfo> mDeviceCapsules;

		std::vector<BallAndSocketJoint> mHostJointsBallAndSocket;
		std::vector<SliderJoint> mHostJointsSlider;
		std::vector<HingeJoint> mHostJointsHinge;
		std::vector<FixedJoint> mHostJointsFixed;
		std::vector<PointJoint> mHostJointsPoint;

	public:
		int m_numOfSamples;
		DArray2D<Vec3f> m_deviceSamples;
		DArray2D<Vec3f> m_deviceNormals;

		std::vector<Vec3f> samples;
		std::vector<Vec3f> normals;

		int getSamplingPointSize() { return m_numOfSamples; }

		DArray2D<Vec3f> getSamples() { return m_deviceSamples; }
		DArray2D<Vec3f> getNormals() { return m_deviceNormals; }
	};
}