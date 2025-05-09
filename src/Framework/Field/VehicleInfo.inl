#ifndef VEHICLEINFO_SERIALIZATION
#define VEHICLEINFO_SERIALIZATION

#include "Field.h"
#include "VehicleInfo.h"

namespace dyno {

	inline void replaceSpacesWithEquals(std::string& substr) {
		for (char& c : substr) {
			if (c == ' ') {
				c = '='; 
			}
		}
	}

	inline void replaceEqualsWithSpaces(std::string& substr) {
		for (char& c : substr) {
			if (c == '=') {
				c = ' ';
			}
		}
	}

	inline void convertRigidInfoToStr(std::string VarName, std::vector<VehicleRigidBodyInfo> Array, std::string& Str)
	{
		Str.append(VarName + " ");
		for (int i = 0; i < Array.size(); i++)
		{
			auto& it = Array[i];
			std::string tempName= it.shapeName.name;
			replaceSpacesWithEquals(tempName);
			Str.append(tempName + " ");	//1
			Str.append(std::to_string(it.shapeName.rigidBodyId) + " ");		//1
			Str.append(std::to_string(it.meshShapeId) + " ");		//1
			Str.append(std::to_string(static_cast<int>(it.shapeType)) + " ");	//1
			auto& t = it.transform.translation();
			auto& r = it.transform.rotation();
			auto& s = it.transform.scale();
			Str.append(std::to_string(t.x) + " " + std::to_string(t.y) + " " + std::to_string(t.z) + " ");
			Str.append(std::to_string(r(0, 0)) + " " + std::to_string(r(0, 1)) + " " + std::to_string(r(0, 2)) + " ");
			Str.append(std::to_string(r(1, 0)) + " " + std::to_string(r(1, 1)) + " " + std::to_string(r(1, 2)) + " ");
			Str.append(std::to_string(r(2, 0)) + " " + std::to_string(r(2, 1)) + " " + std::to_string(r(2, 2)) + " ");
			Str.append(std::to_string(s.x) + " " + std::to_string(s.y) + " " + std::to_string(s.z) + " ");

			Str.append(std::to_string(it.Offset.x) + " " + std::to_string(it.Offset.y) + " " + std::to_string(it.Offset.z) + " ");
			Str.append(std::to_string(it.mHalfLength.x) + " " + std::to_string(it.mHalfLength.y) + " " + std::to_string(it.mHalfLength.z) + " ");
			Str.append(std::to_string(it.radius) + " ");

			Str.append(std::to_string(it.tet[0].x) + " " + std::to_string(it.tet[0].y) + " " + std::to_string(it.tet[0].z) + " ");
			Str.append(std::to_string(it.tet[1].x) + " " + std::to_string(it.tet[1].y) + " " + std::to_string(it.tet[1].z) + " ");
			Str.append(std::to_string(it.tet[2].x) + " " + std::to_string(it.tet[2].y) + " " + std::to_string(it.tet[2].z) + " ");
			Str.append(std::to_string(it.tet[3].x) + " " + std::to_string(it.tet[3].y) + " " + std::to_string(it.tet[3].z) + " ");

			Str.append(std::to_string(it.capsuleLength) + " ");
			Str.append(std::to_string(static_cast<int>(it.motion)) + " ");
			Str.append(std::to_string(it.mDensity) + " ");
			Str.append(std::to_string(int(it.rigidGroup)) + " ");

			if (i != Array.size() - 1)
			{
				Str.append(" \n");
			}
		}
		Str.append(" ");
	}

	inline void convertJointInfoToStr(std::string VarName, std::vector<VehicleJointInfo> Array, std::string& Str)
	{
		Str.append(VarName + " ");
		for (int i = 0; i < Array.size(); i++)
		{
			auto& it = Array[i];

			Str.append(std::to_string(static_cast<int>(it.mJointType)) + " ");
			std::string tempName = it.mRigidBodyName_1.name;
			replaceSpacesWithEquals(tempName);
			Str.append(tempName + " ");	//1
			Str.append(std::to_string(it.mRigidBodyName_1.rigidBodyId) + " ");	
			tempName = it.mRigidBodyName_2.name;
			replaceSpacesWithEquals(tempName);
			Str.append(tempName + " ");	//1
			Str.append(std::to_string(it.mRigidBodyName_2.rigidBodyId) + " ");
			Str.append(std::to_string(int(it.mUseMoter)) + " ");
			Str.append(std::to_string(int(it.mUseRange)) + " ");
			Str.append(std::to_string(int(it.mAnchorPoint[0])) + " " + std::to_string(int(it.mAnchorPoint[1])) + " " + std::to_string(int(it.mAnchorPoint[2])) + " ");
			Str.append(std::to_string(int(it.mMin)) + " ");
			Str.append(std::to_string(int(it.mMax)) + " ");
			Str.append(std::to_string(int(it.mMoter)) + " ");
			Str.append(std::to_string(int(it.mAxis[0])) + " " + std::to_string(int(it.mAxis[1])) + " " + std::to_string(int(it.mAxis[2])) + " ");

			if (i != Array.size() - 1)
			{
				Str.append(" \n");
			}
		}
		Str.append(" ");
	}

	template<>
	inline std::string FVar<VehicleBind>::serialize()
	{

		std::string finalText;
		//serialize Array
		finalText.append(std::to_string(this->getValue().mVehicleRigidBodyInfo.size()) + " ");
		finalText.append(std::to_string(this->getValue().mVehicleJointInfo.size()) + " ");
		finalText.append("\nVehicleRigidBodyInfo \n");
		convertRigidInfoToStr("", this->getValue().mVehicleRigidBodyInfo, finalText);
		finalText.append("\nVehicleJointInfo \n");
		convertJointInfoToStr("", this->getValue().mVehicleJointInfo, finalText);

		std::stringstream ss;
		ss << finalText;

		return ss.str();
	}

	template<>
	inline bool FVar<VehicleBind>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		auto field = std::make_shared<VehicleBind>();
		std::vector<VehicleRigidBodyInfo> rigids;
		std::vector<VehicleJointInfo> joints;

		int dataID = 0;
		bool isRigid = false;
		bool isJoint = false;

		int arrayID = -1;
		int dId = -1;;

		while (ss >> substr)
		{
			std::cout << substr << "\n";
			if (substr == " " || substr == "\n")
				continue;

			{
				if (!isRigid && !isJoint)
				{
					switch (dataID)
					{
					case 0:
						if (std::isdigit(substr[0]))
						{
							rigids.resize(std::stoi(substr));
						}

						break;
					case 1:
						if (std::isdigit(substr[0]))
						{
							joints.resize(std::stoi(substr));
						}
						break;
					default:
						break;
					}
				}
			}

			{
				if (isRigid) 
				{
					arrayID = dataID / 42;
					dId = dataID % 42;
				}
				if (arrayID < rigids.size()&&isRigid)
				{
					switch (dId)
					{
					case 0:
						replaceEqualsWithSpaces(substr);
						rigids[arrayID].shapeName.name = substr;
						break;
					case 1:
						rigids[arrayID].shapeName.rigidBodyId = std::stoi(substr);
						break;
					case 2:
						rigids[arrayID].meshShapeId = std::stoi(substr);
						break;
					case 3:
						rigids[arrayID].shapeType = static_cast<ConfigShapeType>(std::stoi(substr));
						break;
					case 4:
						rigids[arrayID].transform.translation().x = std::stod(substr);
						break;
					case 5:
						rigids[arrayID].transform.translation().y = std::stod(substr);
						break;
					case 6:
						rigids[arrayID].transform.translation().z = std::stod(substr);
						break;
					case 7:
						rigids[arrayID].transform.rotation()(0, 0) = std::stod(substr);
						break;
					case 8:
						rigids[arrayID].transform.rotation()(0, 1) = std::stod(substr);
						break;
					case 9:
						rigids[arrayID].transform.rotation()(0, 2) = std::stod(substr);
						break;
					case 10:
						rigids[arrayID].transform.rotation()(1, 0) = std::stod(substr);
						break;
					case 11:
						rigids[arrayID].transform.rotation()(1, 1) = std::stod(substr);
						break;
					case 12:
						rigids[arrayID].transform.rotation()(1, 2) = std::stod(substr);
						break;
					case 13:
						rigids[arrayID].transform.rotation()(2, 0) = std::stod(substr);
						break;
					case 14:
						rigids[arrayID].transform.rotation()(2, 1) = std::stod(substr);
						break;
					case 15:
						rigids[arrayID].transform.rotation()(2, 2) = std::stod(substr);
						break;
					case 16:
						rigids[arrayID].transform.scale().x = std::stod(substr);
						break;
					case 17:
						rigids[arrayID].transform.scale().y = std::stod(substr);
						break;
					case 18:
						rigids[arrayID].transform.scale().z = std::stod(substr);
						break;
					case 19:
						rigids[arrayID].Offset.x = std::stod(substr);
						break;
					case 20:
						rigids[arrayID].Offset.y = std::stod(substr);
						break;
					case 21:
						rigids[arrayID].Offset.z = std::stod(substr);
						break;
					case 22:
						rigids[arrayID].mHalfLength.x = std::stod(substr);
						break;
					case 23:
						rigids[arrayID].mHalfLength.y = std::stod(substr);
						break;
					case 24:
						rigids[arrayID].mHalfLength.z = std::stod(substr);
						break;
					case 25:
						rigids[arrayID].radius = std::stod(substr);
						break;
					case 26:
						rigids[arrayID].tet[0].x = std::stod(substr);
						break;
					case 27:
						rigids[arrayID].tet[0].y = std::stod(substr);
						break;
					case 28:
						rigids[arrayID].tet[0].z = std::stod(substr);
						break;
					case 29:
						rigids[arrayID].tet[1].x = std::stod(substr);
						break;
					case 30:
						rigids[arrayID].tet[1].y = std::stod(substr);
						break;
					case 31:
						rigids[arrayID].tet[1].z = std::stod(substr);
						break;
					case 32:
						rigids[arrayID].tet[2].x = std::stod(substr);
						break;
					case 33:
						rigids[arrayID].tet[2].y = std::stod(substr);
						break;
					case 34:
						rigids[arrayID].tet[2].z = std::stod(substr);
						break;
					case 35:
						rigids[arrayID].tet[3].x = std::stod(substr);
						break;
					case 36:
						rigids[arrayID].tet[3].y = std::stod(substr);
						break;
					case 37:
						rigids[arrayID].tet[3].z = std::stod(substr);
						break;
					case 38:
						rigids[arrayID].capsuleLength = std::stod(substr);
						break;
					case 39:
						rigids[arrayID].motion = static_cast<ConfigMotionType>(std::stoi(substr));
						break;
					case 40:
						rigids[arrayID].mDensity = std::stoi(substr);
						break;
					case 41:
						rigids[arrayID].rigidGroup = std::stoi(substr);
						break;

					default:
						break;
					}
					arrayID = -1;
					dId = -1;
				}

				if (substr == "VehicleRigidBodyInfo")
				{
					dataID = -1;
					isRigid = true;
					isJoint = false;
				}
				
			}
			
			{
				if (isJoint)
				{
					arrayID = dataID / 16;
					dId = dataID % 16;
				}

				if (arrayID < joints.size() && isJoint)
				{
					switch (dId)
					{
					case 0:
						joints[arrayID].mJointType = static_cast<ConfigJointType>(std::stoi(substr));
						break;
					case 1:
						replaceEqualsWithSpaces(substr);
						joints[arrayID].mRigidBodyName_1.name = substr;
						break;
					case 2:
						joints[arrayID].mRigidBodyName_1.rigidBodyId = std::stoi(substr);
						break;
					case 3:
						replaceEqualsWithSpaces(substr);
						joints[arrayID].mRigidBodyName_2.name = substr;
						break;
					case 4:
						joints[arrayID].mRigidBodyName_2.rigidBodyId = std::stoi(substr);
						break;
					case 5:
						joints[arrayID].mUseMoter = bool(std::stoi(substr));
						break;
					case 6:
						joints[arrayID].mUseRange = bool(std::stoi(substr));
						break;
					case 7:
						joints[arrayID].mAnchorPoint.x = std::stod(substr);
						break;
					case 8:
						joints[arrayID].mAnchorPoint.y = std::stod(substr);
						break;
					case 9:
						joints[arrayID].mAnchorPoint.z = std::stod(substr);
						break;
					case 10:
						joints[arrayID].mMin = std::stod(substr);
						break;
					case 11:
						joints[arrayID].mMax = std::stod(substr);
						break;
					case 12:
						joints[arrayID].mMoter = std::stod(substr);
						break;
					case 13:
						joints[arrayID].mAxis.x = std::stod(substr);
						break;
					case 14:
						joints[arrayID].mAxis.y = std::stod(substr);
						break;
					case 15:
						joints[arrayID].mAxis.z = std::stod(substr);
						break;

					default:
						break;
					}
				}

				if (substr == "VehicleJointInfo")
				{
					dataID = -1;
					isJoint = true;
					isRigid = false;
				}
			}
			
			dataID++;
		}

		field->mVehicleRigidBodyInfo = rigids;
		field->mVehicleJointInfo = joints;
		this->setValue(*field);

		return true;
	}

	inline void convertAnimationJointConfigToStr(Animation2JointConfig& bind, std::string& Str)
	{
		Str.append(bind.JointName + " ");
		Str.append(std::to_string(bind.JointId) + " ");
		Str.append(std::to_string(bind.Axis) + " ");
		Str.append(std::to_string(bind.Intensity) + " ");
	}

	
	template<>
	inline std::string FVar<std::vector<Animation2JointConfig>>::serialize()
	{
		std::string finalText;
		//serialize Array
		auto values = this->getValue();
		finalText.append(std::to_string(values.size()) + " ");
		for (auto it : values)
		{
			convertAnimationJointConfigToStr(it, finalText);
			finalText.append("\n");
		}

		return finalText;
	}
	template<>
	inline bool FVar<std::vector<Animation2JointConfig>>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		std::vector<Animation2JointConfig> animBinds;


		int strID = -1;

		int arrayID = -1;
		int dataID = -1;

		while (ss >> substr)
		{
			std::cout << substr << "\n";
			if (substr == " " || substr == "\n")
				continue;

			strID++;

			switch (strID)
			{
			case 0:
				if (std::isdigit(substr[0]))
					animBinds.resize(std::stoi(substr));

				break;

			}

			arrayID = (strID - 1) / 4;
			dataID = (strID - 1) % 4;

			switch (dataID)
			{

			case 0:
				animBinds[arrayID].JointName = substr;
				break;

			case 1:
				animBinds[arrayID].JointId = std::stoi(substr);
				break;

			case 2:
				animBinds[arrayID].Axis = std::stoi(substr);
				break;

			case 3:
				animBinds[arrayID].Intensity = std::stod(substr);
				break;
			}

			
		}

		this->setValue(animBinds);
		return true;
	}

	template class FVar<std::vector<Animation2JointConfig>>;
}

#endif // !VEHICLEINFO_SERIALIZATION