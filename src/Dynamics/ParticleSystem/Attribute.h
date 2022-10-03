#pragma once
#include "Platform.h"

namespace dyno 
{
	/*!
	*	\class	Attribute
	*	\brief	particle attribute 0x00000000: [31-30]material; [29]motion; [28]Dynamic; [27-8]undefined yet, for future use; [7-0]correspondding to the id of a fluid phase in multiphase fluid or an object in a multibody system
	*/
	class Attribute
	{
	public:
		DYN_FUNC Attribute() { 
			m_tag = 0; 
			this->setDynamic();
			this->setObjectId(OBJECTID_INVALID);
		}

		DYN_FUNC ~Attribute() {};

		enum MaterialType
		{
			MATERIAL_MASK = 0xC0000000,
			MATERIAL_FLUID = 0x00000000,
			MATERIAL_RIGID = 0x40000000,
			MATERIAL_ELASTIC = 0x80000000,
			MATERIAL_PLASTIC = 0xC0000000
		};

		enum KinematicType
		{
			KINEMATIC_MASK = 0x30000000,
			KINEMATIC_FIXED = 0x00000000,
			KINEMATIC_PASSIVE = 0x10000000,
			KINEMATIC_POSITIVE = 0x20000000
		};

		enum ObjectID
		{
			OBJECTID_MASK = 0x000000FF,
			OBJECTID_INVALID = 0x000000FF,
		};

		DYN_FUNC inline void setMaterialType(MaterialType type) { m_tag = ((~MATERIAL_MASK) & m_tag) | type; }
		DYN_FUNC inline void setKinematicType(KinematicType type) { m_tag = ((~KINEMATIC_MASK) & m_tag) | type; }
		DYN_FUNC inline void setObjectId(unsigned id) { m_tag = ((~OBJECTID_MASK) & m_tag) | ((OBJECTID_MASK)& id); }

		DYN_FUNC inline MaterialType materialType() { return (MaterialType)(m_tag&MATERIAL_MASK); }
		DYN_FUNC inline KinematicType kinematicType() { return (KinematicType)(m_tag&KINEMATIC_MASK); }

		DYN_FUNC inline bool isFluid() { return MaterialType::MATERIAL_FLUID == materialType(); }
		DYN_FUNC inline bool isRigid() { return MaterialType::MATERIAL_RIGID == materialType(); }
		DYN_FUNC inline bool isElastic() { return MaterialType::MATERIAL_ELASTIC == materialType(); }
		DYN_FUNC inline bool isPlastic() { return MaterialType::MATERIAL_PLASTIC == materialType(); }

		DYN_FUNC inline void setFluid() { setMaterialType(MaterialType::MATERIAL_FLUID); }
		DYN_FUNC inline void setRigid() { setMaterialType(MaterialType::MATERIAL_RIGID); }
		DYN_FUNC inline void setElastic() { setMaterialType(MaterialType::MATERIAL_ELASTIC); }
		DYN_FUNC inline void setPlastic() { setMaterialType(MaterialType::MATERIAL_PLASTIC); }

		DYN_FUNC inline bool isFixed() { return KinematicType::KINEMATIC_FIXED == kinematicType(); }
		DYN_FUNC inline bool isPassive() { return KinematicType::KINEMATIC_PASSIVE == kinematicType(); }
		DYN_FUNC inline bool isDynamic() { return KinematicType::KINEMATIC_POSITIVE == kinematicType(); }

		DYN_FUNC inline void setFixed() { setKinematicType(KinematicType::KINEMATIC_FIXED); }
		DYN_FUNC inline void setPassive() { setKinematicType(KinematicType::KINEMATIC_PASSIVE); }
		DYN_FUNC inline void setDynamic() { setKinematicType(KinematicType::KINEMATIC_POSITIVE); }

		DYN_FUNC inline unsigned objectId() { return (unsigned)(m_tag&OBJECTID_MASK); }

	private:
		uint m_tag;
	};
}

