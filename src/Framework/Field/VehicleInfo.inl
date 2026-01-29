#ifndef VEHICLEINFO_SERIALIZATION
#define VEHICLEINFO_SERIALIZATION

#include "Field.h"
#include "VehicleInfo.h"
#include "tinygltf/json.hpp"

namespace dyno {

    using json = nlohmann::json;

    // to_json/from_json for Vec3
    inline void to_json(json& j, const Vec3f& v) {
        j = json::array({ v.x, v.y, v.z });
    }
    inline void from_json(const json& j, Vec3f& v) {
        v.x = j.at(0).get<float>();
        v.y = j.at(1).get<float>();
        v.z = j.at(2).get<float>();
    }

    // to_json/from_json for NameId
    inline void to_json(json& j, const Name_Shape& n) {
        j = json{ {"name", n.name}, {"rigidBodyId", n.rigidBodyId} };
    }

    inline void from_json(const json& j, Name_Shape& n) {
        n.name = j.at("name").get<std::string>();
        n.rigidBodyId = j.at("rigidBodyId").get<int>();
    }

    // to_json/from_json for Transform
    inline void to_json(json& j, const Transform3f& t) {
        j = json{
            {"translation", t.translation()},
            {"rotation", {
                {t.rotation()(0,0), t.rotation()(0,1), t.rotation()(0,2)},
                {t.rotation()(1,0), t.rotation()(1,1), t.rotation()(1,2)},
                {t.rotation()(2,0), t.rotation()(2,1), t.rotation()(2,2)}
            }},
            {"scale", t.scale()}
        };
    }

    inline void from_json(const json& j, Transform3f& t) {
        t.translation() = j.at("translation").get<Vec3f>();
        auto rot = j.at("rotation");
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                t.rotation()(i,k) = rot.at(i).at(k).get<double>();
        t.scale() = j.at("scale").get<Vec3f>();
    }

    // to_json/from_json for VehicleRigidBodyInfo
    inline void to_json(json& j, const VehicleRigidBodyInfo& v) {
        j = json{
            {"shapeName", v.shapeName},
            {"meshShapeId", v.meshShapeId},
            {"shapeType", static_cast<int>(v.shapeType)},
            {"translation", v.transform.translation()},
            {"rotation", {
                {v.transform.rotation()(0,0), v.transform.rotation()(0,1), v.transform.rotation()(0,2)},
                {v.transform.rotation()(1,0), v.transform.rotation()(1,1), v.transform.rotation()(1,2)},
                {v.transform.rotation()(2,0), v.transform.rotation()(2,1), v.transform.rotation()(2,2)}
            }},
            {"scale", v.transform.scale()},
            {"Offset", v.Offset},
            {"mHalfLength", v.mHalfLength},
            {"radius", v.radius},
            {"tet", {
                {v.tet[0].x, v.tet[0].y, v.tet[0].z},
                {v.tet[1].x, v.tet[1].y, v.tet[1].z},
                {v.tet[2].x, v.tet[2].y, v.tet[2].z},
                {v.tet[3].x, v.tet[3].y, v.tet[3].z}
            }},
            {"capsuleLength", v.capsuleLength},
            {"motion", static_cast<int>(v.motion)},
            {"mDensity", v.mDensity},
            {"rigidGroup", v.rigidGroup}
        };
    }
    inline void from_json(const json& j, VehicleRigidBodyInfo& v) {
        v.shapeName = j.at("shapeName").get<Name_Shape>();
        v.meshShapeId = j.at("meshShapeId").get<int>();
        v.shapeType = static_cast<ConfigShapeType>(j.at("shapeType").get<int>());
        v.transform.translation() = j.at("translation").get<Vec3f>();
        auto rot = j.at("rotation");
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                v.transform.rotation()(i,k) = rot.at(i).at(k).get<double>();
        v.transform.scale() = j.at("scale").get<Vec3f>();
        v.Offset = j.at("Offset").get<Vec3f>();
        v.mHalfLength = j.at("mHalfLength").get<Vec3f>();
        v.radius = j.at("radius").get<double>();
        auto tet = j.at("tet");
        for (int i = 0; i < 4; ++i) {
            v.tet[i].x = tet.at(i).at(0).get<float>();
            v.tet[i].y = tet.at(i).at(1).get<float>();
            v.tet[i].z = tet.at(i).at(2).get<float>();
        }
        v.capsuleLength = j.at("capsuleLength").get<double>();
        v.motion = static_cast<ConfigMotionType>(j.at("motion").get<int>());
        v.mDensity = j.at("mDensity").get<int>();
        v.rigidGroup = j.at("rigidGroup").get<int>();
    }

    // to_json/from_json for VehicleJointInfo
    inline void to_json(json& j, const VehicleJointInfo& v) {
        j = json{
            {"jointType", static_cast<int>(v.mJointType)},
            {"rigidBodyName_1", v.mRigidBodyName_1.name},
            {"rigidBodyId_1", v.mRigidBodyName_1.rigidBodyId},
            {"rigidBodyName_2", v.mRigidBodyName_2.name},
            {"rigidBodyId_2", v.mRigidBodyName_2.rigidBodyId},
            {"useMoter", v.mUseMoter},
            {"useRange", v.mUseRange},
            {"anchorPoint", {v.mAnchorPoint[0], v.mAnchorPoint[1], v.mAnchorPoint[2]}},
            {"min", v.mMin},
            {"max", v.mMax},
            {"moter", v.mMoter},
            {"axis", {v.mAxis[0], v.mAxis[1], v.mAxis[2]}}
        };
    }

    inline void from_json(const json& j, VehicleJointInfo& v) {
        v.mJointType = static_cast<ConfigJointType>(j.at("jointType").get<int>());
        v.mRigidBodyName_1.name = j.at("rigidBodyName_1").get<std::string>();
        v.mRigidBodyName_1.rigidBodyId = j.at("rigidBodyId_1").get<int>();
        v.mRigidBodyName_2.name = j.at("rigidBodyName_2").get<std::string>();
        v.mRigidBodyName_2.rigidBodyId = j.at("rigidBodyId_2").get<int>();
        v.mUseMoter = j.at("useMoter").get<bool>();
        v.mUseRange = j.at("useRange").get<bool>();
        auto anchor = j.at("anchorPoint");
        v.mAnchorPoint[0] = anchor.at(0).get<float>();
        v.mAnchorPoint[1] = anchor.at(1).get<float>();
        v.mAnchorPoint[2] = anchor.at(2).get<float>();
        v.mMin = j.at("min").get<double>();
        v.mMax = j.at("max").get<double>();
        v.mMoter = j.at("moter").get<double>();
        auto axis = j.at("axis");
        v.mAxis[0] = axis.at(0).get<float>();
        v.mAxis[1] = axis.at(1).get<float>();
        v.mAxis[2] = axis.at(2).get<float>();
    }

    // to_json/from_json for VehicleBind
    inline void to_json(json& j, const MultiBodyBind& v) {
        j = json{
            {"VehicleRigidBodyInfo", v.mVehicleRigidBodyInfo},
            {"VehicleJointInfo", v.mVehicleJointInfo}
        };
    }
    inline void from_json(const json& j, MultiBodyBind& v) {
        v.mVehicleRigidBodyInfo = j.at("VehicleRigidBodyInfo").get<std::vector<VehicleRigidBodyInfo>>();
        v.mVehicleJointInfo = j.at("VehicleJointInfo").get<std::vector<VehicleJointInfo>>();
    }


    template<>
    inline std::string FVar<MultiBodyBind>::serialize()
    {
        json j = this->getValue();
        return j.dump(4); 
    }

    template<>
    inline bool FVar<MultiBodyBind>::deserialize(const std::string& str){
        try {
            json j = json::parse(str);
            auto value = j.get<MultiBodyBind>();
            this->setValue(value);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "deserialize error: " << e.what() << std::endl;
            return false;
        }
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