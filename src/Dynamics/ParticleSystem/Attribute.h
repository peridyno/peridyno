#pragma once
#include <cuda_runtime.h>
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
			this->SetDynamic();
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
			OBJECTID_MASK = 0x000000FF
		};

		enum PartID
		{
			PART_MASK = 0x0000FF00
		};

		DYN_FUNC inline void SetMaterialType(MaterialType type) { m_tag = ((~MATERIAL_MASK) & m_tag) | type; }
		DYN_FUNC inline void SetKinematicType(KinematicType type) { m_tag = ((~KINEMATIC_MASK) & m_tag) | type; }
		DYN_FUNC inline void SetObjectId(unsigned id) { m_tag = ((~OBJECTID_MASK) & m_tag) | id; }

		DYN_FUNC inline MaterialType GetMaterialType() { return (MaterialType)(m_tag&MATERIAL_MASK); }
		DYN_FUNC inline KinematicType GetKinematicType() { return (KinematicType)(m_tag&KINEMATIC_MASK); }

		DYN_FUNC inline bool IsFluid() { return MaterialType::MATERIAL_FLUID == GetMaterialType(); }
		DYN_FUNC inline bool IsRigid() { return MaterialType::MATERIAL_RIGID == GetMaterialType(); }
		DYN_FUNC inline bool IsElastic() { return MaterialType::MATERIAL_ELASTIC == GetMaterialType(); }
		DYN_FUNC inline bool IsPlastic() { return MaterialType::MATERIAL_PLASTIC == GetMaterialType(); }

		DYN_FUNC inline void SetFluid() { SetMaterialType(MaterialType::MATERIAL_FLUID); }
		DYN_FUNC inline void SetRigid() { SetMaterialType(MaterialType::MATERIAL_RIGID); }
		DYN_FUNC inline void SetElastic() { SetMaterialType(MaterialType::MATERIAL_ELASTIC); }
		DYN_FUNC inline void SetPlastic() { SetMaterialType(MaterialType::MATERIAL_PLASTIC); }

		DYN_FUNC inline bool IsFixed() { return KinematicType::KINEMATIC_FIXED == GetKinematicType(); }
		DYN_FUNC inline bool IsPassive() { return KinematicType::KINEMATIC_PASSIVE == GetKinematicType(); }
		DYN_FUNC inline bool IsDynamic() { return KinematicType::KINEMATIC_POSITIVE == GetKinematicType(); }

		DYN_FUNC inline void SetFixed() { SetKinematicType(KinematicType::KINEMATIC_FIXED); }
		DYN_FUNC inline void SetPassive() { SetKinematicType(KinematicType::KINEMATIC_PASSIVE); }
		DYN_FUNC inline void SetDynamic() { SetKinematicType(KinematicType::KINEMATIC_POSITIVE); }

		DYN_FUNC inline void setObjectId(unsigned id) { m_tag |= id; }
		DYN_FUNC inline unsigned getObjectId() { return (unsigned)(m_tag&OBJECTID_MASK); }

		DYN_FUNC inline void setPartId(unsigned id) { m_tag |= id << 8; }
		DYN_FUNC inline unsigned getPartId() { return (unsigned)(m_tag&PART_MASK) >> 8; }

	private:
		unsigned m_tag;
	};
}

