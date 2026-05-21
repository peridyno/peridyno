/**
 * Copyright 2024 Yuzhong Guo
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
#include "ArticulatedBody.h"

#include "STL/Pair.h"

#include "Field/VehicleInfo.h"
#include "Tuple.h"
#include "Field/FList.h"
#include "Matrix/Matrix3x3.h"

namespace dyno
{


	DECLARE_ENUM(RigidShapeType,
		SHAPE_BOX = 1,
		SHAPE_TET = 2,
		SHAPE_CAPSULE = 4,
		SHAPE_SPHERE = 8,
		SHAPE_TRI = 16,
		SHAPE_COMPOUND = 32,
		SHAPE_Other = 0x80000000
	);

	class ShapeTuple : public Tuple
	{
	public:
		ShapeTuple() {};
		ShapeTuple(ShapeTuple& other)
		{
			this->varShapeType()->setValue(other.varShapeType()->getValue());
			this->varCenter()->setValue(other.varCenter()->getValue());
			this->varRot()->setValue(other.varRot()->getValue());
			this->varDensity()->setValue(other.varDensity()->getValue());
			this->varHalfLength()->setValue(other.varHalfLength()->getValue());
			this->varRadius()->setValue(other.varRadius()->getValue());
			this->varCapsuleLength()->setValue(other.varCapsuleLength()->getValue());
			this->varTet()->assign(other.varTet());
		}
		//Shape:
		DEF_ENUM(RigidShapeType, ShapeType, RigidShapeType::SHAPE_CAPSULE, "");
		DEF_VAR(Vec3f, Center, Vec3f(0.0f, 0.0f, 0.0f), "");
		DEF_VAR(Quat<Real>, Rot, Quat<Real>(), "");
		DEF_VAR(Real, Density, 0.0f, "");
		DEF_VAR(Vec3f, HalfLength, Vec3f(0.0f, 0.0f, 0.0f), "");	// if(type == Box);	
		DEF_VAR(Real, Radius, 0.0f, "");							//	if(type == Sphere);  if(type == Capsule);
		DEF_VAR(Real, CapsuleLength, 0.0f, "");						// if(type == Capsule);
		DEF_LIST(Vec3f, Tet, "");									
	};

	DECLARE_ENUM(RigidMotionType,
		RIGID_Static = 0,
		RIGID_Kinematic = 1,
		RIGID_Dynamic = 2,
		RIGID_NonRotatable = 3,
		RIGID_NonGravitative = 4
	);
	DECLARE_ENUM(RigidCollisionMask,
		RIGID_AllObjects = 0xFFFFFFFF,
		RIGID_BoxExcluded = 0xFFFFFFFE,
		RIGID_TetExcluded = 0xFFFFFFFD,
		RIGID_CapsuleExcluded = 0xFFFFFFFA,
		RIGID_SphereExcluded = 0xFFFFFFF7,
		RIGID_BoxOnly = 0x00000001,
		RIGID_TetOnly = 0x00000002,
		RIGID_CapsuleOnly = 0x00000004,
		RIGID_SphereOnly = 0x00000008,
		RIGID_Disabled = 0x00000000
	);

	class RigidBodyTuple : public Tuple
	{
	public:
		RigidBodyTuple() {
		};
		RigidBodyTuple(std::string shapeName, int rigidBodyId = -1)
		{
			this->varShapeName()->setValue(shapeName);
			this->varRigidBodyId()->setValue(rigidBodyId);
		}

		RigidBodyTuple(std::string shapeName, int rigidBodyId,int visualShapeId, RigidShapeType type, Real density = 100)
		{
			this->varShapeName()->setValue(shapeName);
			this->varRigidBodyId()->setValue(rigidBodyId);
			this->varVisualShapeIds()->pushBack(visualShapeId);
			ShapeTuple shape;
			shape.varShapeType()->setCurrentKey(type);
			shape.varDensity()->setValue(density);
			this->varShapeConfigs()->pushBack(shape);
		}

		RigidBodyTuple& operator=(RigidBodyTuple& other) {
			this->varShapeName()->setValue(other.varShapeName()->getValue());
			this->varRigidBodyId()->setValue(other.varRigidBodyId()->getValue());
			this->varAngel()->setValue(other.varAngel()->getValue());
			this->varLinearVelocity()->setValue(other.varLinearVelocity()->getValue());
			this->varAngularVelocity()->setValue(other.varAngularVelocity()->getValue());
			this->varPosition()->setValue(other.varPosition()->getValue());
			this->varOffset()->setValue(other.varOffset()->getValue());
			this->varInertia()->setValue(other.varInertia()->getValue());
			this->varFriction()->setValue(other.varFriction()->getValue());
			this->varRestitution()->setValue(other.varRestitution()->getValue());
			this->varMotionType()->setValue(other.varMotionType()->getValue());
			this->varShapeType()->setValue(other.varShapeType()->getValue());
			this->varCollisionMask()->setValue(other.varCollisionMask()->getValue());
			this->varConfigGroup()->setValue(other.varConfigGroup()->getValue());
			this->varVisualShapeIds()->assign(other.varVisualShapeIds());
			this->varShapeConfigs()->assign(other.varShapeConfigs());

			return *this;
		}

		RigidBodyTuple(RigidBodyTuple& other) {
			varShapeName()->setValue(other.varShapeName()->getValue());
			varRigidBodyId()->setValue(other.varRigidBodyId()->getValue());
			varAngel()->setValue(other.varAngel()->getValue());
			varLinearVelocity()->setValue(other.varLinearVelocity()->getValue());
			varAngularVelocity()->setValue(other.varAngularVelocity()->getValue());
			varPosition()->setValue(other.varPosition()->getValue());
			varOffset()->setValue(other.varOffset()->getValue());
			varInertia()->setValue(other.varInertia()->getValue());
			varFriction()->setValue(other.varFriction()->getValue());
			varRestitution()->setValue(other.varRestitution()->getValue());
			varMotionType()->setValue(other.varMotionType()->getValue());
			varShapeType()->setValue(other.varShapeType()->getValue());
			varCollisionMask()->setValue(other.varCollisionMask()->getValue());
			varConfigGroup()->setValue(other.varConfigGroup()->getValue());
			varVisualShapeIds()->assign(other.varVisualShapeIds());
			varShapeConfigs()->assign(other.varShapeConfigs());
		}
		////Deep copy
		//RigidBodyTuple& operator=(RigidBodyTuple& other) {
		//	//this->varBoolean()->setValue(other.varBoolean()->getValue());
		//	//this->varInt()->setValue(other.varInt()->getValue());
		//	//this->varFloat()->setValue(other.varFloat()->getValue());
		//	//this->varVector()->setValue(other.varVector()->getValue());

		//	//this->varVec3fTupleArray()->assign(other.varVec3fTupleArray());

		//	return *this;
		//}

		DEF_VAR(std::string, ShapeName,"", "");
		DEF_VAR(int, RigidBodyId,-1, "");

		DEF_VAR(Quat<Real>,Angel, Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f),"");
		DEF_VAR(Vec3f, LinearVelocity, Vec3f(0.0f),"");
		DEF_VAR(Vec3f, AngularVelocity, Vec3f(0.0f),"");
		DEF_VAR(Vec3f, Position, Vec3f(std::nanf("")),"");
		DEF_VAR(Vec3f, Offset, Vec3f(0.0f),"");
		DEF_VAR(Mat3f, Inertia, Mat3f(0.0f),"");
		DEF_VAR(Real, Friction, -1.0f,"");
		DEF_VAR(Real, Restitution, 0.0f,"");
		
		DEF_ENUM(RigidMotionType, MotionType, RigidMotionType::RIGID_Dynamic, "RigidMotionType");

		DEF_ENUM(RigidShapeType, ShapeType, RigidShapeType::SHAPE_Other, "RigidMotionType");
		
		DEF_ENUM(RigidCollisionMask, CollisionMask, RigidCollisionMask::RIGID_AllObjects, "RigidMotionType");

		DEF_VAR(int, ConfigGroup, 0, "");
		DEF_LIST(int, VisualShapeIds, "");
		DEF_LIST(ShapeTuple, ShapeConfigs, "");

		bool isValid() { return this->varShapeConfigs()->size() > 0; }
		bool isValidPosition() { return std::isnan(this->varPosition()->getValue().x * this->varPosition()->getValue().y * this->varPosition()->getValue().z); }
	};

	DECLARE_ENUM(JointType,
		JOINT_BallAndSocket = 1,
		JOINT_Slider = 2,
		JOINT_Hinge = 4,
		JOINT_Fixed = 8,
		JOINT_Point = 16,
		JOINT_DistanceJoint = 32,
		JOINT_OtherJoint = 0x80000000
	);

	class MultiBodyJointTuple : public Tuple
	{
	public:
		MultiBodyJointTuple() {};

		MultiBodyJointTuple(std::string AName, int ARigidId, std::string BName, int BRigidId, JointType type, Vec3f Axi = Vec3f(1, 0, 0), Vec3f Point = Vec3f(0), bool Moter = false, Real moter = 0, bool Range = false, Real min = 0, Real max = 0)
		{
			this->varAShapeName()->setValue(AName);
			this->varARigidBodyId()->setValue(ARigidId);
			this->varBShapeName()->setValue(BName);
			this->varBRigidBodyId()->setValue(BRigidId);
			this->varMoter()->setValue(Moter);
			this->varUseMoter()->setValue(Moter!=0);
			this->varUseRange()->setValue(Range);
			this->varAnchorPoint()->setValue(Point);
			this->varRange()->setValue(Vec2f(min,max));
			this->varMoter()->setValue(moter);
			this->varAxis()->setValue(Axi);
			auto s = this->varJointType()->getValue();
			this->varJointType()->setCurrentKey(type);
		}

		MultiBodyJointTuple(MultiBodyJointTuple& other) 
		{
			this->varAShapeName()->setValue(other.varAShapeName()->getValue());
			this->varARigidBodyId()->setValue(other.varARigidBodyId()->getValue());
			this->varBShapeName()->setValue(other.varBShapeName()->getValue());
			this->varBRigidBodyId()->setValue(other.varBRigidBodyId()->getValue());
			this->varAnchorPoint()->setValue(other.varAnchorPoint()->getValue());
			this->varRelativeAnchorPoint()->setValue(other.varRelativeAnchorPoint()->getValue());
			this->varUseMoter()->setValue(other.varUseMoter()->getValue());
			this->varUseRange()->setValue(other.varUseRange()->getValue());
			this->varRange()->setValue(other.varRange()->getValue());
			this->varMoter()->setValue(other.varMoter()->getValue());
			this->varAxis()->setValue(other.varAxis()->getValue());
			this->varQ()->setValue(other.varQ()->getValue());
			this->varR1()->setValue(other.varR1()->getValue());
			this->varR2()->setValue(other.varR2()->getValue());
			this->varDistance()->setValue(other.varDistance()->getValue());
		};
		

		DEF_VAR(std::string, AShapeName, "", "");
		DEF_VAR(int, ARigidBodyId, -1, "");
		DEF_VAR(std::string, BShapeName, "", "");
		DEF_VAR(int, BRigidBodyId, -1, "");
		DEF_VAR(Vec3f, AnchorPoint, Vec3f(0), "");
		DEF_VAR(bool, RelativeAnchorPoint, true, "");
		//SliderJoint  HingeJoint
		DEF_VAR(bool, UseMoter, false, "");
		DEF_VAR(bool, UseRange, false, "");
		//SliderJoint  HingeJoint
		DEF_VAR(Vec2f, Range, Vec2f(0.0f), "");
		DEF_VAR(Real, Moter, 0.0f, "");
		//HingeJoint  SliderJoint 
		DEF_VAR(Vec3f, Axis, Vec3f(0.0f), "");
		//FixedJoint
		DEF_VAR(Quat<Real>, Q, Quat<Real>(), "");
		//distanceJoint  BallAndSocketJoint
		DEF_VAR(Vec3f, R1, Vec3f(0.0f), "");
		DEF_VAR(Vec3f, R2, Vec3f(0.0f), "");
		//distanceJoint
		DEF_VAR(Real, Distance, 0.0f, "");
		DEF_ENUM(JointType, JointType, JointType::JOINT_Hinge, "RigidMotionType");

	};

	class MultiBodyTuple : public Tuple
	{
	public:
		MultiBodyTuple() {};
		MultiBodyTuple(int size) 
		{
			for (int i = 0; i < size; i++)
			{
				this->varRigidBodyConfigs()->pushBack(RigidBodyTuple(std::string("Rigid") + std::to_string(i), i));
			}
		};

		MultiBodyTuple(MultiBodyTuple& other)
		{
			this->varRigidBodyConfigs()->assign(other.varRigidBodyConfigs());
			this->varJointConfigs()->assign(other.varJointConfigs());
		}

		~MultiBodyTuple() {};

		bool isValid() { return this->varRigidBodyConfigs()->size(); }

		DEF_LIST(RigidBodyTuple, RigidBodyConfigs, "");
		DEF_LIST(MultiBodyJointTuple, JointConfigs, "");

	};

	template<typename TDataType>
	class ConfigurableBody : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(ConfigurableBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ConfigurableBody();
		~ConfigurableBody() override;


		//DEF_VAR(MultiBodyBind, Configuration, MultiBodyBind(4), "");

		DEF_TUPLE(MultiBodyTuple, Configuration, "Define a Tuple");

		DEF_VAR(FilePath, LoadConfigPath, FilePath("", "Peridyno Multibody Files (*.pdm)"), "");


	public:
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "Input TextureMesh");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet of the boundary");

	protected:
		void resetStates() override;
		void saveToFile() override;
		void loadFromFile();
		void updateConfig();

	};
}
