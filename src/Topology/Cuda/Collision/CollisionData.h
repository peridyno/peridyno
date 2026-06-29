#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
#define INVLIDA_ID -1
	
	enum CModeMask
	{
		CM_Disabled 			= 0x00000000,		
		CM_OriginDCD_Tet		= 0x00000001,			// Origin DCD for local contact point
		CM_InputSDF_Tet			= 0x00000002,			// Intergated SDF 
		CM_RigidSurface_Tet 	= 0x00000004,			// Rigid Surface 
		CM_TetMesh_Tet			= 0x00000008,			// TetMesh for semi-local contact point
		CM_SurfaceMesh_Tet 		= 0x00000010, 			// Surface triangle mesh for semi-local contact point
		CM_OriginDCD_Sphere		= 0x00000020,
		CM_InputSDF_Sphere		= 0x00000040
	};

	enum BodyType
	{
		Static = 0,
		Kinematic,
		Dynamic,
		NonRotatable,
		NonGravitative,
	};


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
		CT_SURFACE,
		CT_VERTEX_SDF,
		CT_VERTEX_FACE,
		CT_EDGE_EDGE,
		CT_UNKNOWN
	};

	struct BoxInfo
	{
		BoxInfo()
		{
			center = Vector<Real, 3>(0.0f, 0.0f, 0.0f);
			halfLength = Vector<Real, 3>(1.0f, 1.0f, 1.0f);
			rot = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f);
		}

		Vector<Real, 3> center;
		Vector<Real, 3> halfLength;

		Quat<Real> rot;
	};

	struct SphereInfo
	{
		SphereInfo()
		{
			center = Vector<Real, 3>(0.0f, 0.0f, 0.0f);
			radius = 1.0;
			rot = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f);
		}
		Quat<Real> rot;
		Vector<Real, 3> center;
		Real radius;
	};

	struct TetInfo
	{
		TetInfo()
		{
			v[0] = Vec3f(0);
			v[1] = Vec3f(1, 0, 0);
			v[2] = Vec3f(0, 1, 0);
			v[3] = Vec3f(0, 0, 1);
		}

		Vector<Real, 3> v[4];
	};

	struct MedialConeInfo {
		MedialConeInfo()
		{
			v[0] = Vec3f(0);
			v[1] = Vec3f(1, 0, 0);
			radius[0] = 1.0;
			radius[1] = 1.0;

		}
		Vector<Real, 3> v[2];
		Real radius[2];
	};

	struct MedialSlabInfo {
		MedialSlabInfo()
		{
			v[0] = Vec3f(0);
			v[1] = Vec3f(1, 0, 0);
			v[2] = Vec3f(0, 1, 0);
			radius[0] = 1.0;
			radius[1] = 1.0;
			radius[2] = 1.0;
		}
		Vector<Real, 3> v[3];
		Real radius[3];
	};

	struct CapsuleInfo
	{
		CapsuleInfo()
		{
			center = Vec3f(0.0f, 0.0f, 0.0f);
			rot = Quat1f(0.0f, 0.0f, 0.0f, 1.0f);
			radius = 1.0f;
			halfLength = 1.0f;
		}
		Quat<Real> rot;
		Vector<Real, 3> center;
		Real halfLength;
		Real radius;
	};

	template<typename Real>
	class TContact
	{
	public:
		Vector<Real, 3> position;			// World coordinate of contact
		Real penetration;			// Depth of penetration from collision whose value is assumed to be negative when interpenetration occurs
	};

	template<typename Real>
	struct TManifold
	{
	public:
		Vector<Real, 3> normal;				// on B
		TContact<Real> contacts[8];
		int contactCount = 0;

		DYN_FUNC void pushContact(const Vector<Real, 3>& pos, const Real& dep)
		{
			if (contactCount >= 8) return;
			contacts[contactCount].position = pos;
			contacts[contactCount].penetration = dep;
			contactCount++;
		}

		DYN_FUNC void pushContactReWrite(const Vector<Real, 3>& pos, const Real& dep)
		{
			contacts[0].position = pos;
			contacts[0].penetration = dep;
			contactCount = 1;
		}

		DYN_FUNC void pushContact(const TContact<Real>& contact)
		{
			if (contactCount >= 8) return;
			contacts[contactCount] = contact;
			contactCount++;
		}
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
		
		int localId1;
		int localId2;
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

	struct CacheContact {
		unsigned long long sortKey;
		Vec3f localPos;
		Real lambda1;
		Real lambda2;
		Real lambda3;
	};
}