
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
			rigidId = Id;
		}

		//Var
		std::string name = "";
		int rigidId = -1;

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

		Vec3f halfLength = Vec3f(1);	// if(type == Box);
		
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
			JointName1 = Name1;
			JointName2 = Name2;
			useMoter = Moter;
			useRange = Range;
			anchorPoint = Point;
			d_min = min;
			d_max = max;
			v_moter = moter;
			Axis = Axi;
			type = typeIn;
		}

		ConfigJointType type;
		Name_Shape JointName1;
		Name_Shape JointName2;
		//************************  Joint  ************************:
		//Vec2i Joint_Actor = Vec2i(-1, -1);//update ElementType bodyType1;ElementType bodyType2;PdActor* actor1 = nullptr;PdActor* actor2 = nullptr;
		bool useMoter = false;
		bool useRange = false;
		// anchor point in body local space
		Vector<Real, 3> anchorPoint = Vec3f(0);
		Real d_min = 0;
		Real d_max = 0;
		Real v_moter = 0;
		Vector<Real, 3> Axis = Vector<Real, 3>(1,0,0);
	};

	class VehicleBind 
	{
	public:
		VehicleBind() {};
		VehicleBind(int size) 
		{
			vehicleRigidBodyInfo.resize(size);
			vehicleJointInfo.resize(size);
		}
		~VehicleBind() 
		{
			vehicleRigidBodyInfo.clear();
			vehicleJointInfo.clear();
		}

		bool isValid() 
		{
			return vehicleRigidBodyInfo.size();
		}

		std::vector<VehicleRigidBodyInfo> vehicleRigidBodyInfo;
		std::vector<VehicleJointInfo> vehicleJointInfo;

	};


}

