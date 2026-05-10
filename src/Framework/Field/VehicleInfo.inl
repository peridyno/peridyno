#ifndef VEHICLEINFO_SERIALIZATION
#define VEHICLEINFO_SERIALIZATION

#include "Field.h"
#include "VehicleInfo.h"
#include "tinygltf/json.hpp"
#include "Quat.h"
#include "Matrix.h"

namespace dyno {

    using json = nlohmann::json;

    inline void to_json(json& j, const Vector<Real, 3>& v) {
        j = json::array({ v[0], v[1], v[2] });
    }

    inline void from_json(const json& j, Vector<Real, 3>& v) {
        v[0] = (j.at(0));
        v[1] = (j.at(1));
        v[2] = (j.at(2));
    }

    inline void to_json(json& j, const Quat<Real>& q) {
        j = json::array({ q.x, q.y, q.z, q.w });
    }

    inline void from_json(const json& j, Quat<Real>& q) {
        q.x = (j.at(0));
        q.y = (j.at(1));
        q.z = (j.at(2));
        q.w = (j.at(3));
    }

    inline void to_json(json& j, const SquareMatrix<Real, 3>& m) {
        j = json::array({
            {m(0,0), m(0,1), m(0,2)},
            {m(1,0), m(1,1), m(1,2)},
            {m(2,0), m(2,1), m(2,2)}
            });
    }

    inline void from_json(const json& j, SquareMatrix<Real, 3>& m) {
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                m(i, k) = (j.at(i).at(k));
            }
        }
    }

    inline void to_json(json& j, const NameRigidID& n) {
        j = json{ {"name", n.name}, {"rigidBodyId", n.rigidBodyId} };
    }
    inline void from_json(const json& j, NameRigidID& n) {
        n.name = j.at("name").get<std::string>();
        n.rigidBodyId = j.at("rigidBodyId").get<int>();
    }

    inline void to_json(json& j, const ShapeConfig& s) {
        j = json{
            {"shapeType", static_cast<int>(s.shapeType)},
            {"center", s.center},
            {"rot", s.rot},
            {"density", s.density},
            {"halfLength", s.halfLength},
            {"radius", s.radius},
            {"tet", s.tet},
            {"capsuleLength", s.capsuleLength}
        };
    }
    inline void from_json(const json& j, ShapeConfig& s) {
        s.shapeType = static_cast<ConfigShapeType>(j.at("shapeType").get<int>());
        s.center = j.at("center").get<Vec3f>();
        s.rot = j.at("rot").get<Quat<Real>>();
        s.density = j.at("density").get<Real>();
        s.halfLength = j.at("halfLength").get<Vec3f>();
        s.radius = j.at("radius").get<float>();
        s.tet = j.at("tet").get<std::vector<Vec3f>>();
        s.capsuleLength = j.at("capsuleLength").get<float>();
    }

    inline void to_json(json& j, const RigidBodyConfig& r) {
        j = json{
            {"shapeName", r.shapeName},
            {"angle", r.angle},
            {"linearVelocity", r.linearVelocity},
            {"angularVelocity", r.angularVelocity},
            {"position", r.position},
            {"offset", r.offset},
            {"ConfigGroup", r.ConfigGroup},
            {"inertia", r.inertia},
            {"friction", r.friction},
            {"restitution", r.restitution},
            {"motionType", static_cast<int>(r.motionType)},
            {"shapeType", static_cast<int>(r.shapeType)},
            {"collisionMask", static_cast<uint32_t>(r.collisionMask)},
            {"visualShapeIds", r.visualShapeIds},
            {"shapeConfigs", r.shapeConfigs}
        };
    }
    inline void from_json(const json& j, RigidBodyConfig& r) {
        r.shapeName = j.at("shapeName").get<NameRigidID>();
        r.angle = j.at("angle").get<Quat<Real>>();
        r.linearVelocity = j.at("linearVelocity").get<Vector<Real, 3>>();
        r.angularVelocity = j.at("angularVelocity").get<Vector<Real, 3>>();
        r.position = j.at("position").get<Vector<Real, 3>>();
        r.offset = j.at("offset").get<Vector<Real, 3>>();
        r.ConfigGroup = j.at("ConfigGroup").get<int>();
        r.inertia = j.at("inertia").get<SquareMatrix<Real, 3>>();
        r.friction = j.at("friction").get<Real>();
        r.restitution = j.at("restitution").get<Real>();
        r.motionType = static_cast<ConfigMotionType>(j.at("motionType").get<int>());
        r.shapeType = static_cast<ConfigShapeType>(j.at("shapeType").get<int>());
        r.collisionMask = static_cast<ConfigCollisionMask>(j.at("collisionMask").get<uint32_t>());
        r.visualShapeIds = j.at("visualShapeIds").get<std::vector<int>>();
        r.shapeConfigs = j.at("shapeConfigs").get<std::vector<ShapeConfig>>();
    }

    inline void to_json(json& j, const MultiBodyJointConfig& mj) {
        j = json{
            {"mJointType", static_cast<int>(mj.mJointType)},
            {"mRigidBodyName_1", mj.mRigidBodyName_1},
            {"mRigidBodyName_2", mj.mRigidBodyName_2},
            {"mUseMoter", mj.mUseMoter},
            {"mUseRange", mj.mUseRange},
            {"mAnchorPoint", mj.mAnchorPoint},
            {"mMin", mj.mMin},
            {"mMax", mj.mMax},
            {"mMoter", mj.mMoter},
            {"mAxis", mj.mAxis},
            {"q", mj.q},
            {"r1", mj.r1},
            {"r2", mj.r2},
            {"distance", mj.distance}
        };
    }
    inline void from_json(const json& j, MultiBodyJointConfig& mj) {
        mj.mJointType = static_cast<ConfigJointType>(j.at("mJointType").get<int>());
        mj.mRigidBodyName_1 = j.at("mRigidBodyName_1").get<NameRigidID>();
        mj.mRigidBodyName_2 = j.at("mRigidBodyName_2").get<NameRigidID>();
        mj.mUseMoter = j.at("mUseMoter").get<bool>();
        mj.mUseRange = j.at("mUseRange").get<bool>();
        mj.mAnchorPoint = j.at("mAnchorPoint").get<Vec3f>();
        mj.mMin = j.at("mMin").get<Real>();
        mj.mMax = j.at("mMax").get<Real>();
        mj.mMoter = j.at("mMoter").get<Real>();
        mj.mAxis = j.at("mAxis").get<Vec3f>();
        mj.q = j.at("q").get<Quat<Real>>();
        mj.r1 = j.at("r1").get<Vec3f>();
        mj.r2 = j.at("r2").get<Vec3f>();
        mj.distance = j.at("distance").get<Real>();
    }

    inline void to_json(json& j, const MultiBodyBind& mb) {
        j = json{
            {"rigidBodyConfigs", mb.rigidBodyConfigs},
            {"jointConfigs", mb.jointConfigs}
        };
    }
    inline void from_json(const json& j, MultiBodyBind& mb) {
        mb.rigidBodyConfigs = j.at("rigidBodyConfigs").get<std::vector<RigidBodyConfig>>();
        mb.jointConfigs = j.at("jointConfigs").get<std::vector<MultiBodyJointConfig>>();
    }


    template<>
    inline std::string FVar<MultiBodyBind>::serialize() {
        json j = this->getValue();
        return j.dump(4); 
    }

    template<>
    inline bool FVar<MultiBodyBind>::deserialize(const std::string& str) {
        try {
            if (str.empty()) {
                std::cerr << "deserialize error: empty string" << std::endl;
                return false;
            }
            json j = json::parse(str);
            MultiBodyBind value = j.get<MultiBodyBind>();
            this->setValue(value);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "deserialize error: " << e.what() << std::endl;
            return false;
        }
    }


    inline void convertAnimationJointConfigToStr(Animation2JointConfig& bind, std::string& Str) {
        Str.append(bind.JointName + " ");
        Str.append(std::to_string(bind.JointId) + " ");
        Str.append(std::to_string(bind.Axis) + " ");
        Str.append(std::to_string(bind.Intensity) + " ");
    }

    template<>
    inline std::string FVar<std::vector<Animation2JointConfig>>::serialize() {
        std::string finalText;
        auto values = this->getValue();
        finalText.append(std::to_string(values.size()) + " ");
        for (auto& it : values) {
            convertAnimationJointConfigToStr(it, finalText);
            finalText.append("\n");
        }
        return finalText;
    }

    template<>
    inline bool FVar<std::vector<Animation2JointConfig>>::deserialize(const std::string& str) {
        if (str.empty())
            return false;

        std::stringstream ss(str);
        std::string substr;
        std::vector<Animation2JointConfig> animBinds;

        int strID = -1;
        int arrayID = -1;
        int dataID = -1;

        while (ss >> substr) {
            if (substr.empty()) continue;
            strID++;

            if (strID == 0) {
                if (std::isdigit(substr[0])) {
                    try {
                        animBinds.resize(std::stoi(substr));
                    }
                    catch (...) {
                        std::cerr << "deserialize error: invalid array size" << std::endl;
                        return false;
                    }
                }
                continue;
            }

            arrayID = (strID - 1) / 4;
            dataID = (strID - 1) % 4;

            if (arrayID >= (int)animBinds.size()) {
                std::cerr << "deserialize error: data out of range" << std::endl;
                return false;
            }

            switch (dataID) {
            case 0: animBinds[arrayID].JointName = substr; break;
            case 1: animBinds[arrayID].JointId = std::stoi(substr); break;
            case 2: animBinds[arrayID].Axis = std::stoi(substr); break;
            case 3: animBinds[arrayID].Intensity = std::stod(substr); break;
            default: break;
            }
        }

        this->setValue(animBinds);
        return true;
    }

    template class FVar<MultiBodyBind>;
    template class FVar<std::vector<Animation2JointConfig>>;

} // namespace dyno

#endif // !VEHICLEINFO_SERIALIZATION

//
//namespace dyno {
//
//    using json = nlohmann::json;
//
//    inline void to_json(json& j, const Vec3f& v) {
//        j = json::array({ v.x, v.y, v.z });
//    }
//    inline void from_json(const json& j, Vec3f& v) {
//        v.x = j.at(0).get<float>();
//        v.y = j.at(1).get<float>();
//        v.z = j.at(2).get<float>();
//    }
//
//    // Quat<Real> 序列化
//    inline void to_json(json& j, const Quat<Real>& q) {
//        j = json::array({ q.x, q.y, q.z, q.w });
//    }
//    inline void from_json(const json& j, Quat<Real>& q) {
//        q.x = j.at(0).get<Real>();
//        q.y = j.at(1).get<Real>();
//        q.z = j.at(2).get<Real>();
//        q.w = j.at(3).get<Real>();
//    }
//
//    // Vector<Real,3> 序列化
//    inline void to_json(json& j, const Vector<Real, 3>& v) {
//        j = json::array({ v[0], v[1], v[2] });
//    }
//    inline void from_json(const json& j, Vector<Real, 3>& v) {
//        v[0] = j.at(0).get<Real>();
//        v[1] = j.at(1).get<Real>();
//        v[2] = j.at(2).get<Real>();
//    }
//
//    // SquareMatrix<Real,3> 序列化
//    inline void to_json(json& j, const SquareMatrix<Real, 3>& m) {
//        j = json::array({
//            json::array({ m(0,0),m(0,1), m(0,2) }),
//            json::array({ m(1,0), m(1,1), m(1,2) }),
//            json::array({ m(2,0), m(2,1), m(2,2) })
//            });
//    }
//    inline void from_json(const json& j, SquareMatrix<Real, 3>& m) {
//        for (int i = 0; i < 3; ++i)
//            for (int k = 0; k < 3; ++k)
//                m(i,k) = j.at(i).at(k).get<Real>();
//    }
//
//    // NameRigidID 序列化
//    inline void to_json(json& j, const NameRigidID& n) {
//        j = json{ {"name", n.name}, {"rigidBodyId", n.rigidBodyId} };
//    }
//    inline void from_json(const json& j, NameRigidID& n) {
//        n.name = j.at("name").get<std::string>();
//        n.rigidBodyId = j.at("rigidBodyId").get<int>();
//    }
//
//    // ShapeConfig 序列化
//    inline void to_json(json& j, const ShapeConfig& s) {
//        j = json{
//            {"shapeType", static_cast<int>(s.shapeType)},
//            {"center", s.center},
//            {"rot", s.rot},
//            {"density", s.density},
//            {"halfLength", s.halfLength},
//            {"radius", s.radius},
//            {"tet", s.tet},
//            {"capsuleLength", s.capsuleLength}
//        };
//    }
//    inline void from_json(const json& j, ShapeConfig& s) {
//        s.shapeType = static_cast<ConfigShapeType>(j.at("shapeType").get<int>());
//        s.center = j.at("center").get<Vec3f>();
//        s.rot = j.at("rot").get<Quat<Real>>();
//        s.density = j.at("density").get<Real>();
//        s.halfLength = j.at("halfLength").get<Vec3f>();
//        s.radius = j.at("radius").get<float>();
//        s.tet = j.at("tet").get<std::vector<Vec3f>>();
//        s.capsuleLength = j.at("capsuleLength").get<float>();
//    }
//
//    // RigidBodyConfig 序列化
//    inline void to_json(json& j, const RigidBodyConfig& r) {
//        j = json{
//            {"shapeConfigs", r.shapeConfigs},
//            {"visualShapeIds", r.visualShapeIds},
//            {"shapeName", r.shapeName},
//            {"angle", r.angle},
//            {"linearVelocity", r.linearVelocity},
//            {"angularVelocity", r.angularVelocity},
//            {"position", r.position},
//            {"offset", r.offset},
//            {"inertia", r.inertia},
//            {"bodyId", r.bodyId},
//            {"friction", r.friction},
//            {"restitution", r.restitution},
//            {"motionType", static_cast<int>(r.motionType)},
//            {"shapeType", static_cast<int>(r.shapeType)},
//            {"collisionMask", static_cast<unsigned int>(r.collisionMask)}
//        };
//    }
//    inline void from_json(const json& j, RigidBodyConfig& r) {
//        r.shapeConfigs = j.at("shapeConfigs").get<std::vector<ShapeConfig>>();
//        r.visualShapeIds = j.at("visualShapeIds").get<std::vector<int>>();
//        r.shapeName = j.at("shapeName").get<NameRigidID>();
//        r.angle = j.at("angle").get<Quat<Real>>();
//        r.linearVelocity = j.at("linearVelocity").get<Vector<Real, 3>>();
//        r.angularVelocity = j.at("angularVelocity").get<Vector<Real, 3>>();
//        r.position = j.at("position").get<Vector<Real, 3>>();
//        r.offset = j.at("offset").get<Vector<Real, 3>>();
//        r.inertia = j.at("inertia").get<SquareMatrix<Real, 3>>();
//        r.bodyId = j.at("bodyId").get<unsigned int>();
//        r.friction = j.at("friction").get<Real>();
//        r.restitution = j.at("restitution").get<Real>();
//        r.motionType = static_cast<ConfigMotionType>(j.at("motionType").get<int>());
//        r.shapeType = static_cast<ConfigShapeType>(j.at("shapeType").get<int>());
//        r.collisionMask = static_cast<ConfigCollisionMask>(j.at("collisionMask").get<unsigned int>());
//    }
//
//    // MultiBodyJointConfig 序列化
//    inline void to_json(json& j, const MultiBodyJointConfig& m) {
//        j = json{
//            {"mJointType", static_cast<int>(m.mJointType)},
//            {"mRigidBodyName_1", m.mRigidBodyName_1},
//            {"mRigidBodyName_2", m.mRigidBodyName_2},
//            {"mUseMoter", m.mUseMoter},
//            {"mUseRange", m.mUseRange},
//            {"mAnchorPoint", m.mAnchorPoint},
//            {"mMin", m.mMin},
//            {"mMax", m.mMax},
//            {"mMoter", m.mMoter},
//            {"mAxis", m.mAxis}
//        };
//    }
//    inline void from_json(const json& j, MultiBodyJointConfig& m) {
//        m.mJointType = static_cast<ConfigJointType>(j.at("mJointType").get<int>());
//        m.mRigidBodyName_1 = j.at("mRigidBodyName_1").get<NameRigidID>();
//        m.mRigidBodyName_2 = j.at("mRigidBodyName_2").get<NameRigidID>();
//        m.mUseMoter = j.at("mUseMoter").get<bool>();
//        m.mUseRange = j.at("mUseRange").get<bool>();
//        m.mAnchorPoint = j.at("mAnchorPoint").get<Vector<Real, 3>>();
//        m.mMin = j.at("mMin").get<Real>();
//        m.mMax = j.at("mMax").get<Real>();
//        m.mMoter = j.at("mMoter").get<Real>();
//        m.mAxis = j.at("mAxis").get<Vector<Real, 3>>();
//    }
//
//    // MultiBodyBind 序列化
//    inline void to_json(json& j, const MultiBodyBind& m) {
//        j = json{
//            {"rigidBodyConfigs", m.rigidBodyConfigs},
//            {"jointConfigs", m.jointConfigs}
//        };
//    }
//    inline void from_json(const json& j, MultiBodyBind& m) {
//        m.rigidBodyConfigs = j.at("rigidBodyConfigs").get<std::vector<RigidBodyConfig>>();
//        m.jointConfigs = j.at("jointConfigs").get<std::vector<MultiBodyJointConfig>>();
//    }
//
//    // FVar<MultiBodyBind> 序列化示例（假设你有FVar模板）
//    template<>
//    inline std::string FVar<MultiBodyBind>::serialize() {
//        json j = this->getValue();
//        return j.dump(4);
//    }
//
//    template<>
//    inline bool FVar<MultiBodyBind>::deserialize(const std::string& str) {
//        try {
//            json j = json::parse(str);
//            auto value = j.get<MultiBodyBind>();
//            this->setValue(value);
//            return true;
//        }
//        catch (const std::exception& e) {
//            std::cerr << "deserialize error: " << e.what() << std::endl;
//            return false;
//        }
//    }
//
//
//	inline void convertAnimationJointConfigToStr(Animation2JointConfig& bind, std::string& Str)
//	{
//		Str.append(bind.JointName + " ");
//		Str.append(std::to_string(bind.JointId) + " ");
//		Str.append(std::to_string(bind.Axis) + " ");
//		Str.append(std::to_string(bind.Intensity) + " ");
//	}
//
//	
//	template<>
//	inline std::string FVar<std::vector<Animation2JointConfig>>::serialize()
//	{
//		std::string finalText;
//		//serialize Array
//		auto values = this->getValue();
//		finalText.append(std::to_string(values.size()) + " ");
//		for (auto it : values)
//		{
//			convertAnimationJointConfigToStr(it, finalText);
//			finalText.append("\n");
//		}
//
//		return finalText;
//	}
//	template<>
//	inline bool FVar<std::vector<Animation2JointConfig>>::deserialize(const std::string& str)
//	{
//		if (str.empty())
//			return false;
//
//		std::stringstream ss(str);
//		std::string substr;
//
//		std::vector<Animation2JointConfig> animBinds;
//
//
//		int strID = -1;
//
//		int arrayID = -1;
//		int dataID = -1;
//
//		while (ss >> substr)
//		{
//			std::cout << substr << "\n";
//			if (substr == " " || substr == "\n")
//				continue;
//
//			strID++;
//
//			switch (strID)
//			{
//			case 0:
//				if (std::isdigit(substr[0]))
//					animBinds.resize(std::stoi(substr));
//
//				break;
//
//			}
//
//			arrayID = (strID - 1) / 4;
//			dataID = (strID - 1) % 4;
//
//			switch (dataID)
//			{
//
//			case 0:
//				animBinds[arrayID].JointName = substr;
//				break;
//
//			case 1:
//				animBinds[arrayID].JointId = std::stoi(substr);
//				break;
//
//			case 2:
//				animBinds[arrayID].Axis = std::stoi(substr);
//				break;
//
//			case 3:
//				animBinds[arrayID].Intensity = std::stod(substr);
//				break;
//			}
//
//			
//		}
//
//		this->setValue(animBinds);
//		return true;
//	}
//
//	template class FVar<std::vector<Animation2JointConfig>>;
//}
//
//#endif // !VEHICLEINFO_SERIALIZATION