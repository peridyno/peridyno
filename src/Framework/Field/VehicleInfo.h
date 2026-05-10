
#pragma once
#include <vector>
#include <memory>
#include <string>

#include "Vector.h"
#include "Matrix.h"
#include "OBase.h"
#include "Module/InputModule.h"


namespace dyno {

	enum ConfigMotionType
	{
		CONFIG_Static = 0,
		CONFIG_Kinematic = 1,
		CONFIG_Dynamic = 2,
		CONFIG_NonRotatable = 3,
		CONFIG_NonGravitative = 4
	};

	enum ConfigShapeType
	{
		CONFIG_BOX = 1,
		CONFIG_TET = 2,
		CONFIG_CAPSULE = 4,
		CONFIG_SPHERE = 8,
		CONFIG_TRI = 16,
		CONFIG_COMPOUND = 32,
		CONFIG_Other = 0x80000000
	};

	enum ConfigJointType
	{
		CONFIG_BallAndSocket = 1,
		CONFIG_Slider = 2,
		CONFIG_Hinge = 4,
		CONFIG_Fixed = 8,
		CONFIG_Point = 16,
		CONFIG_DistanceJoint = 32,
		CONFIG_OtherJoint = 0x80000000
	};

	enum ConfigCollisionMask
	{
		CONFIG_AllObjects = 0xFFFFFFFF,
		CONFIG_BoxExcluded = 0xFFFFFFFE,
		CONFIG_TetExcluded = 0xFFFFFFFD,
		CONFIG_CapsuleExcluded = 0xFFFFFFFA,
		CONFIG_SphereExcluded = 0xFFFFFFF7,
		CONFIG_BoxOnly = 0x00000001,
		CONFIG_TetOnly = 0x00000002,
		CONFIG_CapsuleOnly = 0x00000004,
		CONFIG_SphereOnly = 0x00000008,
		CONFIG_Disabled = 0x00000000
	};

	struct NameRigidID
	{
		//Func
		NameRigidID() {};
		NameRigidID(std::string n, int Id = -1)
		{
			name = n;
			rigidBodyId = Id;
		}

		//Var
		std::string name = "";
		int rigidBodyId = -1;

	};

	struct ShapeConfig
	{
		ShapeConfig() {};

		//Shape:	
		ConfigShapeType shapeType = ConfigShapeType::CONFIG_CAPSULE;
		Vector<Real, 3> center = Vector<Real, 3>(0);
		Quat<Real> rot = Quat<Real>();
		Real density = 100;

		Vector<Real, 3> halfLength = Vector<Real, 3>(0);											// if(type == Box);	
		float radius = 0;														//	if(type == Sphere);  if(type == Capsule);
		std::vector<Vector<Real, 3>> tet = { Vector<Real, 3>(0),Vector<Real, 3>(0),Vector<Real, 3>(0),Vector<Real, 3>(0) };	//	if(type == Tet);
		float capsuleLength = 0;												// if(type == Capsule);
	};

	struct RigidBodyConfig
	{
		RigidBodyConfig()
		{
		}


		RigidBodyConfig(Vector<Real, 3> p, Quat<Real> q = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f))
		{
			position = p;
			angle = q;
		}

		RigidBodyConfig(NameRigidID name, int visualShapeId, ConfigShapeType type, Vector<Real,3> position , Quat<Real> angle = Quat<Real>(), Real density = 100) 
		{
			shapeName = name;
			visualShapeIds.push_back(visualShapeId);
			ShapeConfig shape;
			shape.shapeType = type;
			shape.density = density;
			shapeConfigs.push_back(shape);
			this->position = position;
			this->angle = angle;
		}

		RigidBodyConfig(NameRigidID name, int visualShapeId, ConfigShapeType type, Real density = 100)
		{
			shapeName = name;
			visualShapeIds.push_back(visualShapeId);
			ShapeConfig shape;
			shape.shapeType = type;
			shape.density = density;
			shapeConfigs.push_back(shape);
			this->position = position;
			this->angle = angle;
		}

		void bindShapeConfig(
			const ShapeConfig& shapeConfig
		)
		{
			shapeConfigs.push_back(shapeConfig);
		}

		void bindVisualShapeConfig(
			int visualShapeId
		)
		{
			visualShapeIds.push_back(visualShapeId);
		}

		NameRigidID shapeName = NameRigidID("");
		Quat<Real> angle = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f);
		Vector<Real, 3> linearVelocity = Vector<Real, 3>(0.0f);
		Vector<Real, 3> angularVelocity = Vector<Real, 3>(0.0f);
		Vector<Real, 3> position = Vector<Real, 3>(std::nanf(""));
		Vector<Real, 3> offset = Vector<Real, 3>(0.0f);
		SquareMatrix<Real, 3> inertia = SquareMatrix<Real, 3>(0.0f);;
		Real friction = -1.0f;
		Real restitution = 0.0f;;
		ConfigMotionType motionType = ConfigMotionType::CONFIG_Dynamic;
		ConfigShapeType shapeType = ConfigShapeType::CONFIG_Other;
		ConfigCollisionMask collisionMask = ConfigCollisionMask::CONFIG_AllObjects;

		int ConfigGroup = 0;
		std::vector<int> visualShapeIds;
		std::vector<ShapeConfig> shapeConfigs;

		bool isValid() const { return shapeConfigs.size() > 0; }
		bool isValidPosition() const { return std::isnan(position.x * position.y * position.z); }
	};


	struct MultiBodyJointConfig
	{
		MultiBodyJointConfig() {};
		MultiBodyJointConfig(NameRigidID Name1, NameRigidID Name2, ConfigJointType typeIn, Vector<Real, 3> Axi = Vec3f(1, 0, 0), Vector<Real, 3> Point = Vec3f(0), bool Moter = false, Real moter = 0, bool Range = false, Real min = 0, Real max = 0);

		ConfigJointType mJointType;	
		NameRigidID mRigidBodyName_1;
		NameRigidID mRigidBodyName_2;

		Vector<Real, 3> mAnchorPoint = Vec3f(0);
		bool relativeAnchorPoint = true;
		//SliderJoint  HingeJoint
		bool mUseMoter = false;
		bool mUseRange = false;
		Real mMin = 0;
		Real mMax = 0;
		Real mMoter = 0;
		//HingeJoint  SliderJoint 
		Vector<Real, 3> mAxis = Vector<Real, 3>(1, 0, 0);

		//FixedJoint
		Quat<Real> q = Quat<Real>();

		//distanceJoint  BallAndSocketJoint
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;//FVar<Vec3f>

		//distanceJoint
		Real distance;
	};


	class MultiBodyBind
	{
	public:
		MultiBodyBind() {};
		MultiBodyBind(int size);
		~MultiBodyBind();

		bool isValid() { return rigidBodyConfigs.size(); }

		std::vector<RigidBodyConfig> rigidBodyConfigs;
		std::vector<MultiBodyJointConfig> jointConfigs;

	};

	class Animation2JointConfig
	{
	public:

		Animation2JointConfig() {}
		Animation2JointConfig(std::string name, int id, uint axis)
		{
			this->JointName = name;
			this->JointId = id;
			this->Axis = axis;
		}
		Animation2JointConfig(std::string name, int id, uint axis, float intensity)
		{
			this->JointName = name;
			this->JointId = id;
			this->Axis = axis;
			this->Intensity = intensity;
		}
		std::string JointName;
		int JointId = -1;
		uint Axis = 0;
		float Intensity = 1;
	};

	struct HingeAction
	{
		HingeAction(int jointId, float value)
		{
			this->joint = jointId;
			this->value = value;
		}
		int joint = -1;
		float value = -1;
	};

	class Key2HingeConfig
	{
	public:

		Key2HingeConfig() {}
		~Key2HingeConfig() {}


		void addMap(PKeyboardType key, int jointId, float value)
		{
			auto& vec = key2Hinge[key];

			for (auto& action : vec)
			{
				if (action.joint == jointId)
				{
					action.value = value;
					return;
				}
			}

			vec.emplace_back(jointId, value);
		}

		std::map<PKeyboardType, std::vector<HingeAction>> key2Hinge;


	};

}
#include "VehicleInfo.inl"

