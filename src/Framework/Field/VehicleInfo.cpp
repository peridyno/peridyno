#include "VehicleInfo.h"


namespace dyno 
{

	MultiBodyJointConfig::MultiBodyJointConfig(
		NameRigidID Name1,
		NameRigidID Name2,
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

	MultiBodyBind::MultiBodyBind(int size)
	{
		rigidBodyConfigs.resize(size);
		for (int i = 0; i < rigidBodyConfigs.size(); i++)
		{
			rigidBodyConfigs[i].shapeName = NameRigidID(std::string("Rigid") + std::to_string(i), i);
			rigidBodyConfigs[i].visualShapeIds.push_back(i);
		}
	}

	MultiBodyBind::~MultiBodyBind()
	{
		rigidBodyConfigs.clear();
		jointConfigs.clear();
	}

}

