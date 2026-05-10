#include "MultiBodySaveHelper.h"

namespace dyno
{

	template<typename TDataType>
	void MultiBodySaveHelper<TDataType>::initialShapeConfig(
		ShapeConfig& shapeRigid,
		ConfigShapeType type,
		Vec3f center,
		Quat<Real> rot,
		Vec3f halfLength,
		Real radius,
		Real capsuleLength,
		Real density,
		std::vector<Vec3f> tet
	)
	{
		shapeRigid.shapeType = type;
		shapeRigid.center = center;
		shapeRigid.rot = rot;
		shapeRigid.halfLength = halfLength;
		shapeRigid.radius = radius;
		shapeRigid.capsuleLength = capsuleLength;
		shapeRigid.density = density;
		shapeRigid.tet = { Vec3f(0),Vec3f(0),Vec3f(0),Vec3f(0,1,0) };
	}

	template<typename TDataType>
	void MultiBodySaveHelper<TDataType>::initialRigidBodyConfig(
		RigidBodyConfig& configRigid,

		std::string name,
		int rigidbodyID,
		Quat<Real> angle,
		Vector<Real, 3> linearVelocity,
		Vector<Real, 3> angularVelocity,
		Vector<Real, 3> position,
		Vector<Real, 3> offset,
		SquareMatrix<Real, 3> inertia,
		uint bodyId,
		Real friction,
		Real restitution,
		ConfigMotionType motionType,
		ConfigShapeType shapeType,
		ConfigCollisionMask collisionMask,
		std::vector<int> visualShapeIds,
		std::vector<ShapeConfig> shapeConfigs
	)
	{
		configRigid.shapeName = NameRigidID(name + std::to_string(rigidbodyID), rigidbodyID);
		configRigid.angle = angle;
		configRigid.linearVelocity = linearVelocity;
		configRigid.angularVelocity = angularVelocity;
		configRigid.position = position;
		configRigid.offset = offset;
		configRigid.inertia = inertia;
		configRigid.bodyId = bodyId;
		configRigid.friction = friction;
		configRigid.restitution = restitution;
		configRigid.motionType = motionType;
		configRigid.shapeType = shapeType;
		configRigid.collisionMask = collisionMask;
		configRigid.visualShapeIds = visualShapeIds;
		configRigid.shapeConfigs = shapeConfigs;
	}

	template<typename TDataType>
	ConfigMotionType MultiBodySaveHelper<TDataType>::ToConfigMotionType(BodyType bodyType)
	{
		switch (bodyType)
		{
		case Static: return CONFIG_Static;
		case Kinematic: return CONFIG_Kinematic;
		case Dynamic: return CONFIG_Dynamic;
		case NonRotatable: return CONFIG_NonRotatable;
		case NonGravitative: return CONFIG_NonGravitative;
		default: return CONFIG_Dynamic;
		}
	}

	template<typename TDataType>
	ConfigShapeType MultiBodySaveHelper<TDataType>::ToConfigShapeType(ElementType element)
	{
		switch (element)
		{
		case ET_BOX:       return CONFIG_BOX;
		case ET_TET:       return CONFIG_TET;
		case ET_CAPSULE:   return CONFIG_CAPSULE;
		case ET_SPHERE:    return CONFIG_SPHERE;
		case ET_TRI:       return CONFIG_TRI;
		case ET_COMPOUND:  return CONFIG_COMPOUND;
		case ET_Other:     return CONFIG_Other;
		default:           return CONFIG_Other; // Ä¬ÈÏ·µ»Ø
		}
	}

	template<typename TDataType>
	ConfigCollisionMask MultiBodySaveHelper<TDataType>::ToConfigCollisionMask(CollisionMask mask)
	{
		switch (mask)
		{
		case CT_AllObjects:       return CONFIG_AllObjects;
		case CT_BoxExcluded:      return CONFIG_BoxExcluded;
		case CT_TetExcluded:      return CONFIG_TetExcluded;
		case CT_CapsuleExcluded:  return CONFIG_CapsuleExcluded;
		case CT_SphereExcluded:   return CONFIG_SphereExcluded;
		case CT_BoxOnly:          return CONFIG_BoxOnly;
		case CT_TetOnly:          return CONFIG_TetOnly;
		case CT_CapsuleOnly:      return CONFIG_CapsuleOnly;
		case CT_SphereOnly:       return CONFIG_SphereOnly;
		case CT_Disabled:         return CONFIG_Disabled;
		default:                  return CONFIG_AllObjects;
		}
	}


	template<typename TDataType>
	MultiBodyBind MultiBodySaveHelper<TDataType>::getMultiBodyBind(
		const std::vector<RigidBodyInfo>& getRigidBodyStates,
		const std::vector<SphereInfo>& getSpheres,
		const std::vector<BoxInfo>& getBoxes,
		const std::vector<TetInfo>& getTets,
		const std::vector<CapsuleInfo>& getCapsules,
		const std::vector<BallAndSocketJoint>& getJointsBallAndSocket,
		const std::vector<SliderJoint>& getJointsSlider,
		const std::vector<HingeJoint>& getJointsHinge,
		const std::vector<FixedJoint>& getJointsFixed,
		const std::vector<PointJoint>& getJointsPoint,
		const std::vector<Pair<uint, uint>>& getShape2RigidBodyMapping
	)
	{
		int rigidNum = getRigidBodyStates.size();
		int jointNum = getJointsBallAndSocket.size() + getJointsSlider.size() + getJointsHinge.size() + getJointsFixed.size() + getJointsPoint.size();

		MultiBodyBind config;
		config.rigidBodyConfigs.resize(rigidNum);
		config.jointConfigs.resize(jointNum);

		for (size_t rId = 0; rId < rigidNum; rId++)
		{
			auto rigidInfo = getRigidBodyStates[rId];
			RigidBodyConfig& rigidConfig = config.rigidBodyConfigs[rId];

			if (!rigidConfig.isValid())
			{
				initialRigidBodyConfig(
					rigidConfig,
					"Rigid",
					rId,
					rigidInfo.angle,
					rigidInfo.linearVelocity,
					rigidInfo.angularVelocity,
					rigidInfo.position,
					rigidInfo.offset,
					rigidInfo.inertia,
					rigidInfo.bodyId,
					rigidInfo.friction,
					rigidInfo.restitution,
					ToConfigMotionType(rigidInfo.motionType),
					ToConfigShapeType(rigidInfo.shapeType),
					ToConfigCollisionMask(rigidInfo.collisionMask)
				);
			}
		}

		for (size_t i = 0; i < getSpheres.size(); i++)
		{
			auto pair = getShape2RigidBodyMapping[i];
			uint rId = pair.second;
			auto sphereInfo = getSpheres[i];
			auto rigidInfo = getRigidBodyStates[rId];

			ShapeConfig sphereConfig;
			initialShapeConfig(
				sphereConfig,
				ConfigShapeType::CONFIG_SPHERE,
				sphereInfo.center,
				sphereInfo.rot,
				Vec3f(0),
				sphereInfo.radius,
				0,
				rigidInfo.mass / ((4.0 / 3.0) * M_PI * sphereInfo.radius * sphereInfo.radius * sphereInfo.radius)
			);

			config.rigidBodyConfigs[rId].bindShapeConfig(sphereConfig);

		}

		for (size_t i = 0; i < getBoxes.size(); i++)
		{
			auto pair = getShape2RigidBodyMapping[getSpheres.size() + i];
			uint rId = pair.second;
			auto boxInfo = getBoxes[i];

			auto rigidInfo = getRigidBodyStates[rId];

			float lx = 2.0f * boxInfo.halfLength[0];
			float ly = 2.0f * boxInfo.halfLength[1];
			float lz = 2.0f * boxInfo.halfLength[2];

			ShapeConfig boxConfig;
			initialShapeConfig(
				boxConfig,
				ConfigShapeType::CONFIG_BOX,
				boxInfo.center,
				boxInfo.rot,
				boxInfo.halfLength,
				0,
				0,
				rigidInfo.mass / (lx * ly * lz)
			);

			config.rigidBodyConfigs[rId].bindShapeConfig(boxConfig);

		}

		for (size_t i = 0; i < getTets.size(); i++)
		{
			auto pair = getShape2RigidBodyMapping[getSpheres.size() + getBoxes.size() + i];
			uint rId = pair.second;
			auto tetInfo = getTets[i];

			auto rigidInfo = getRigidBodyStates[rId];

			Coord centroid = (tetInfo.v[0] + tetInfo.v[1] + tetInfo.v[2] + tetInfo.v[3]) / 4;

			Coord v0 = tetInfo.v[0] - centroid;
			Coord v1 = tetInfo.v[1] - centroid;
			Coord v2 = tetInfo.v[2] - centroid;
			Coord v3 = tetInfo.v[3] - centroid;

			auto tmpMat = Mat3f(v1 - v0, v2 - v0, v3 - v0);

			Real detJ = abs(tmpMat.determinant());
			Real volume = (1.0 / 6.0) * detJ;
			Real density = rigidInfo.mass / volume;

			std::vector<Vec3f> tet;
			tet.push_back(tetInfo.v[0]);
			tet.push_back(tetInfo.v[1]);
			tet.push_back(tetInfo.v[2]);
			tet.push_back(tetInfo.v[3]);

			ShapeConfig tetConfig;
			initialShapeConfig(
				tetConfig,
				ConfigShapeType::CONFIG_TET,
				Vec3f(0),
				Quat<Real>(),
				Vec3f(0),
				0,
				0,
				density,
				tet
			);

			config.rigidBodyConfigs[rId].bindShapeConfig(tetConfig);
			printf("tet : rigidID - %d\n", int(rId));

		}

		for (size_t i = 0; i < getCapsules.size(); i++)
		{
			auto pair = getShape2RigidBodyMapping[getSpheres.size() + getBoxes.size() + getTets.size() + i];
			uint rId = pair.second;
			auto capsuleInfo = getCapsules[i];
			auto rigidInfo = getRigidBodyStates[rId];

			Real r = capsuleInfo.radius;
			Real h = capsuleInfo.halfLength * 2;
			Real density = rigidInfo.mass / (2.0 / 3.0 * M_PI * r * r * r * 2 + M_PI * r * r * h);

			ShapeConfig capsuleConfig;
			initialShapeConfig(
				capsuleConfig,
				ConfigShapeType::CONFIG_CAPSULE,
				capsuleInfo.center,
				capsuleInfo.rot,
				Vec3f(0),
				capsuleInfo.radius,
				capsuleInfo.halfLength,
				density
			);

			config.rigidBodyConfigs[rId].bindShapeConfig(capsuleConfig);

		}

		return config;
	}

}