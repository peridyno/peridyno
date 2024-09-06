
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
		Static = 0,
		Kinematic = 1,
		Dynamic = 2,
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

	struct VehicleRigidBodyInfo 
	{	
		VehicleRigidBodyInfo() {};

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type) //
		{
			shapeName = name;
			meshShapeId = shapeId;
			shapeType = type;
		};

		VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Transform3f trans) //
		{
			shapeName = name;
			meshShapeId = shapeId;
			shapeType = type;
			transform = trans;
		};

		//Shape:
		Name_Shape shapeName = "";

		int meshShapeId = -1;
		ConfigShapeType shapeType = ConfigShapeType::Capsule;
		Transform3f transform = Transform3f(Vec3f(0), Mat3f::identityMatrix(), Vec3f(1));

		Vec3f Offset = Vec3f(0);
		//if MeshShapeId

		Vec3f mHalfLength = Vec3f(1);	// if(type == Box);
		
		float radius = 1;	//	if(type == Sphere);  if(type == Capsule);

		std::vector<Vec3f> tet = {Vec3f(0),Vec3f(0),Vec3f(0),Vec3f(0,1,0) };	//	if(type == Tet);

		float capsuleLength = 1;	// if(type == Capsule);

		ConfigMotionType motion = ConfigMotionType::Dynamic;

	};



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
		//************************  Joint  ************************:
		//Vec2i Joint_Actor = Vec2i(-1, -1);//update ElementType bodyType1;ElementType bodyType2;PdActor* actor1 = nullptr;PdActor* actor2 = nullptr;
		bool mUseMoter = false;
		bool mUseRange = false;
		// anchor point in body local space
		Vector<Real, 3> mAnchorPoint = Vec3f(0);
		Real mMin = 0;
		Real mMax = 0;
		Real mMoter = 0;
		Vector<Real, 3> mAxis = Vector<Real, 3>(1,0,0);
	};

	class VehicleBind 
	{
	public:
		VehicleBind() {};
		VehicleBind(int size) 
		{
			mVehicleRigidBodyInfo.resize(size);
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

