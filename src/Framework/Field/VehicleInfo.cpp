#include "VehicleInfo.h"


namespace dyno 
{

	VehicleRigidBodyInfo::VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Real density) //
	{
		shapeName = name;
		meshShapeId = shapeId;
		shapeType = type;
		mDensity = density;
	};

	VehicleRigidBodyInfo::VehicleRigidBodyInfo(Name_Shape name, int shapeId, ConfigShapeType type, Transform3f trans, Real density) //
	{
		shapeName = name;
		meshShapeId = shapeId;
		shapeType = type;
		transform = trans;
		mDensity = density;
	};

	VehicleJointInfo::VehicleJointInfo(
		Name_Shape Name1,
		Name_Shape Name2,
		ConfigJointType typeIn,
		Vector<Real, 3> Axi ,
		Vector<Real, 3> Point ,
		bool Moter ,
		Real moter ,
		bool Range ,
		Real min ,
		Real max 
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

	VehicleBind::VehicleBind(int size)
	{
		mVehicleRigidBodyInfo.resize(size);
		for (int i = 0; i < mVehicleRigidBodyInfo.size(); i++)
		{
			mVehicleRigidBodyInfo[i].shapeName = Name_Shape(std::string("Rigid") + std::to_string(i), i);
			mVehicleRigidBodyInfo[i].meshShapeId = i;
		}
		mVehicleJointInfo.resize(size);
	}

	VehicleBind::~VehicleBind()
	{
		mVehicleRigidBodyInfo.clear();
		mVehicleJointInfo.clear();
	}

}

