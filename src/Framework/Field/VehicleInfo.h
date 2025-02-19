
#pragma once
#include <vector>
#include <memory>
#include <string>

#include "Vector.h"
#include "Matrix.h"
#include "OBase.h"


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
		OtherShape = 0x80000000
	};

	enum ConfigJointType
	{
		BallAndSocket = 1,
		Slider = 2,
		Hinge = 3,
		Fixed = 4,
		Point = 5,
		OtherJoint = 0x80000000
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

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type,Real density = 100) //
		{
			shapeName = name;
			meshShapeId = shapeId;
			shapeType = type;
			mDensity = density;
		};

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Transform3f trans, Real density = 100) //
		{
			shapeName = name;
			meshShapeId = shapeId;
			shapeType = type;
			transform = trans;
			mDensity = density;
		};

		//Shape:
		Name_Shape shapeName = Name_Shape("");

		int meshShapeId = -1;
		ConfigShapeType shapeType = ConfigShapeType::Capsule;
		Transform3f transform = Transform3f(Vec3f(0), Mat3f::identityMatrix(), Vec3f(1));
		Vec3f Offset = Vec3f(0);

		Vec3f mHalfLength = Vec3f(1);		// if(type == Box);	
		float radius = 1;					//	if(type == Sphere);  if(type == Capsule);
		std::vector<Vec3f> tet = {Vec3f(0),Vec3f(0),Vec3f(0),Vec3f(0,1,0) };	//	if(type == Tet);
		float capsuleLength = 1;			// if(type == Capsule);
		ConfigMotionType motion = ConfigMotionType::CMT_Dynamic;

		Real mDensity = 100;

		uint rigidGroup = 0;
	};

	/**
	 * @brief The joint information is stored in mVehicleJointInfo.
	 */
	struct VehicleJointInfo
	{
		VehicleJointInfo() {};
		VehicleJointInfo(
			Name_Shape Name1,
			Name_Shape Name2,
			ConfigJointType typeIn,
			Vector<Real, 3> Axi = Vec3f(1, 0, 0),
			Vector<Real, 3> Point = Vec3f(0),
			bool Moter = false,
			Real moter = 0,
			bool Range = false,
			Real min = 0,
			Real max = 0
		) 
		{
			mRigidBodyName_1 = Name1;
			mRigidBodyName_2 = Name2;
			mUseMoter = Moter;
			mUseRange = Range;
			mAnchorPoint = Point;
			mMin = min;
			mMax = max;
			mMoter = moter;
			mAxis = Axi;
			mJointType = typeIn;
		}

		ConfigJointType mJointType;
		Name_Shape mRigidBodyName_1;
		Name_Shape mRigidBodyName_2;
		bool mUseMoter = false;
		bool mUseRange = false;
		Vector<Real, 3> mAnchorPoint = Vec3f(0);
		Real mMin = 0;
		Real mMax = 0;
		Real mMoter = 0;
		Vector<Real, 3> mAxis = Vector<Real, 3>(1,0,0);
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
		VehicleBind(int size) 
		{
			mVehicleRigidBodyInfo.resize(size);
			for (int i = 0; i < mVehicleRigidBodyInfo.size(); i++)
			{
				mVehicleRigidBodyInfo[i].shapeName = Name_Shape(std::string("Rigid") + std::to_string(i),i);
				mVehicleRigidBodyInfo[i].meshShapeId = i;
			}
			mVehicleJointInfo.resize(size);
		}
		~VehicleBind() 
		{
			mVehicleRigidBodyInfo.clear();
			mVehicleJointInfo.clear();
		}

		bool isValid() 
		{
			return mVehicleRigidBodyInfo.size();
		}

		std::vector<VehicleRigidBodyInfo> mVehicleRigidBodyInfo;
		std::vector<VehicleJointInfo> mVehicleJointInfo;

	};


}

