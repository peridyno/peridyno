
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
		CMT_Static = 0,
		CMT_Kinematic = 1,
		CMT_Dynamic = 2,
	};

	enum ConfigShapeType
	{
		Box = 1,
		Tet = 2,
		Capsule = 4,
		Sphere = 8,
		Tri = 16,
		OtherShape = 999
	};

	enum ConfigJointType
	{
		BallAndSocket = 1,
		Slider = 2,
		Hinge = 3,
		Fixed = 4,
		Point = 5,
		OtherJoint = 999
	};

	struct Name_Shape 
	{
		//Func
		Name_Shape() {};
		Name_Shape(std::string n,int Id = -1) 
		{
			name = n;
			rigidBodyId = Id;
		}

		//Var
		std::string name = "";
		int rigidBodyId = -1;

	};

	/**
	 * @brief The Rigid body information is stored in mVehicleJointInfo.
	 */
	struct VehicleRigidBodyInfo 
	{	
		VehicleRigidBodyInfo() {};

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Real density = 100);

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Transform3f trans, Real density = 100);

		//Shape:
		Name_Shape shapeName = Name_Shape("");//2

		int meshShapeId = -1;//1	//3
		ConfigShapeType shapeType = ConfigShapeType::Capsule;//1	//4
		Transform3f transform = Transform3f(Vec3f(0), Mat3f::identityMatrix(), Vec3f(1));//15	//19
		Vec3f Offset = Vec3f(0);//3		//22

		Vec3f mHalfLength = Vec3f(1);//3	//25		// if(type == Box);	
		float radius = 1;//1	//26					//	if(type == Sphere);  if(type == Capsule);
		std::vector<Vec3f> tet = {Vec3f(0),Vec3f(0),Vec3f(0),Vec3f(0,1,0) };//12	//38 	//	if(type == Tet);
		float capsuleLength = 1;//1		//39			// if(type == Capsule);
		ConfigMotionType motion = ConfigMotionType::CMT_Dynamic;//1		//40

		Real mDensity = 100;//1		//41

		uint rigidGroup = 0;//1		//42
	};

	/**
	 * @brief The joint information is stored in mVehicleJointInfo.
	 */
	struct VehicleJointInfo
	{
		VehicleJointInfo() {};
		VehicleJointInfo(Name_Shape Name1,Name_Shape Name2,ConfigJointType typeIn,Vector<Real, 3> Axi = Vec3f(1, 0, 0),Vector<Real, 3> Point = Vec3f(0),bool Moter = false,Real moter = 0,bool Range = false,Real min = 0,Real max = 0);

		ConfigJointType mJointType;	//1
		Name_Shape mRigidBodyName_1;//2
		Name_Shape mRigidBodyName_2;//2
		bool mUseMoter = false;//1
		bool mUseRange = false;//1
		Vector<Real, 3> mAnchorPoint = Vec3f(0);//3
		Real mMin = 0;//1
		Real mMax = 0;//1
		Real mMoter = 0;//1
		Vector<Real, 3> mAxis = Vector<Real, 3>(1,0,0);//3
	};


	/**
	 * @brief The VehicleBind class is used to record information about created rigid bodies and joints.
			  Rigid bodies information is stored in mVehicleRigidBodyInfo.
			  Toints information is stored in mVehicleJointInfo.
	 */
	class VehicleBind 
	{
	public:
		VehicleBind() {};
		VehicleBind(int size);
		~VehicleBind();

		bool isValid() {return mVehicleRigidBodyInfo.size();}

		std::vector<VehicleRigidBodyInfo> mVehicleRigidBodyInfo;
		std::vector<VehicleJointInfo> mVehicleJointInfo;



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

