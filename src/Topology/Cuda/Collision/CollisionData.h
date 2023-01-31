#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
#define INVLIDA_ID -1

	enum CollisionMask
	{
		CT_AllObjects = 0xFFFFFFFF,
		CT_BoxExcluded = 0xFFFFFFFE,
		CT_TetExcluded = 0xFFFFFFFD,
		CT_CapsuleExcluded = 0xFFFFFFFA,
		CT_SphereExcluded = 0xFFFFFFF7,
		CT_BoxOnly = 0x00000001,
		CT_TetOnly = 0x00000002,
		CT_CapsuleOnly = 0x00000004,
		CT_SphereOnly = 0x00000008,
		CT_Disabled = 0x00000000
	};

	enum ContactType
	{
		CT_BOUDNARY = 0,
		CT_INTERNAL,
		CT_NONPENETRATION,
		CT_UNKNOWN
	};

	enum ConstraintType
	{
		CN_NONPENETRATION = 0,
		CN_FRICTION,
		CN_FLUID_STICKINESS,
		CN_FLUID_SLIPINESS,
		CN_FLUID_NONPENETRATION,
		CN_GLOBAL_NONPENETRATION,
		CN_LOACL_NONPENETRATION,
		CN_UNKNOWN
	};

	template<typename Real>
	class TContact
	{
	public:
		Vector<Real, 3> position;			// World coordinate of contact
		Real penetration;			// Depth of penetration from collision
	};

	template<typename Real>
	struct TManifold
	{
	public:
		Vector<Real, 3> normal;				// From A to B
		TContact<Real> contacts[8];
		int contactCount = 0;
	};

	template<typename Real>
	class TContactPair
	{
	public:
		DYN_FUNC TContactPair()
		{
			bodyId1 = bodyId2 = INVLIDA_ID;
			contactType = CT_UNKNOWN;
		};

		DYN_FUNC TContactPair(
			int a, 
			int b, 
			ContactType type, 
			Vector<Real, 3> p1, 
			Vector<Real, 3> p2, 
			Vector<Real, 3> n1, 
			Vector<Real, 3> n2)
		{
			bodyId1 = a;
			bodyId2 = b;
			contactType = type;
			pos1 = p1;
			pos2 = p2;
			normal1 = n1;
			normal2 = n2;
		}

		int bodyId1;
		int bodyId2;
		
// 		int localId1;
// 		int localId2;
// 
// 		int localTag1;
// 		int localTag2;

		Real interpenetration = 0.0f;//inter_dist

		Vector<Real, 3> pos1;
		Vector<Real, 3> pos2;

		Vector<Real, 3> normal1;
		Vector<Real, 3> normal2;

		ContactType contactType;
	};

	template<typename Real>
	class TConstraintPair
	{
	public:
		DYN_FUNC TConstraintPair()
		{
			bodyId1 = bodyId2 = INVLIDA_ID;
			type = CN_UNKNOWN;
		};

		DYN_FUNC TConstraintPair(
			int a,
			int b,
			ConstraintType type,
			Vector<Real, 3> p1,
			Vector<Real, 3> p2,
			Vector<Real, 3> n1,
			Vector<Real, 3> n2)
		{
			bodyId1 = a;
			bodyId2 = b;
			type = type;
			pos1 = p1;
			pos2 = p2;
			normal1 = n1;
			normal2 = n2;
		}

		int bodyId1;
		int bodyId2;

		int localId1;
		int localId2;

		int localTag1;
		int localTag2;

		Real interpenetration = 0.0f;//inter_dist

		Vector<Real, 3> pos1;
		Vector<Real, 3> pos2;

		Vector<Real, 3> normal1;
		Vector<Real, 3> normal2;

		ConstraintType type;
	};
}