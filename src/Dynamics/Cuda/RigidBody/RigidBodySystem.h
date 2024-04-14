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
#include "Topology/Joint.h"
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
	class RigidBodySystem : public Node
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


		RigidBodySystem();
		~RigidBodySystem() override;

		void addBox(
			const BoxInfo& box, 
			const RigidBodyInfo& bodyDef,
			const Real density = Real(0.1));

		void addSphere(
			const SphereInfo& sphere,
			const RigidBodyInfo& bodyDef, 
			const Real density = Real(0.1));

		void addTet(
			const TetInfo& tet,
			const RigidBodyInfo& bodyDef,
			const Real density = Real(0.1));

		void addBallAndSocketJoint(
			const BallAndSocketJoint& joint
		);

		void addSliderJoint(
			const SliderJoint& joint
		);

		void addHingeJoint(
			const HingeJoint& joint
		);

		void addFixedJoint(
			const FixedJoint& joint
		);

		Mat3f pointInertia(Coord v1);

		Real getDt() override { return double(0.001); }

	protected:
		void resetStates() override;

		void updateTopology() override;

	public:
		DEF_VAR(bool, FrictionEnabled, true, "A toggle to control the friction");

		DEF_VAR(bool, GravityEnabled, true, "A toggle to control the gravity");

		DEF_VAR(Real, GravityValue, 9.8, "");

		DEF_VAR(Real, FrictionCoefficient, 0.002, "");

		DEF_VAR(Real, Slop, 0.0001, "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

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

		DEF_ARRAY_STATE(Matrix, InitialInertia, DeviceType::GPU, "Initial inertia matrix");

		DEF_ARRAY_STATE(BallAndSocketJoint, BallAndSocketJoints, DeviceType::GPU, "Ball And Socket Joints");

		DEF_ARRAY_STATE(SliderJoint, SliderJoints, DeviceType::GPU, "Slider Joints");

		DEF_ARRAY_STATE(HingeJoint, HingeJoints, DeviceType::GPU, "Hinge Joints");

		DEF_ARRAY_STATE(FixedJoint, FixedJoints, DeviceType::GPU, "Fixed Joints");

	private:
		std::vector<RigidBodyInfo> mHostRigidBodyStates;

		std::vector<SphereInfo> mHostSpheres;
		std::vector<BoxInfo> mHostBoxes;
		std::vector<TetInfo> mHostTets;

		DArray<RigidBodyInfo> mDeviceRigidBodyStates;

		DArray<SphereInfo> mDeviceSpheres;
		DArray<BoxInfo> mDeviceBoxes;
		DArray<TetInfo> mDeviceTets;

		std::vector<BallAndSocketJoint> mHostJointsBallAndSocket;
		std::vector<SliderJoint> mHostJointsSlider;
		std::vector<HingeJoint> mHostJointsHinge;
		std::vector<FixedJoint> mHostJointsFixed;

	public:
		int m_numOfSamples;
		DArray2D<Vec3f> m_deviceSamples;
		DArray2D<Vec3f> m_deviceNormals;

		std::vector<Vec3f> samples;
		std::vector<Vec3f> normals;

		int getSamplingPointSize() { return m_numOfSamples; }

		DArray2D<Vec3f> getSamples() { return m_deviceSamples; }
		DArray2D<Vec3f> getNormals() { return m_deviceNormals; }

		//float m_damping = 0.9f;

		float m_yaw;
		float m_pitch;
		float m_roll;
		float m_recoverSpeed;
	};
}