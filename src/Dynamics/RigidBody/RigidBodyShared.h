#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
	enum MotionType
	{
		MT_Static = 0,
		MT_Kinematic,
		MT_Dynamic,
	};

	enum CollisionMask
	{
		CT_AllObjects = 0xFFFFFFFF,
		CT_BoxExcluded = 0xFFFFFFFE,
		CT_TetExcluded = 0xFFFFFFFD,
		CT_SegmentExcluded = 0xFFFFFFFA,
		CT_SphereExcluded = 0xFFFFFFF7,
		CT_BoxOnly = 0x00000001,
		CT_TetOnly = 0x00000002,
		CT_SegmentOnly = 0x00000004,
		CT_SphereOnly = 0x00000008,
		CT_Disabled = 0x00000000
	};

	enum ShapeType
	{
		ST_Box = 1,
		ST_Tet = 2,
		ST_Capsule = 4,
		ST_Sphere = 8,
		ST_Other = 0x80000000
	};

	struct BoxInfo
	{
		BoxInfo()
		{
			center = Vec3f(0.0f, 0.0f, 0.0f);
			halfLength = Vec3f(1.0f, 1.0f, 1.0f);
			rot = Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
		}

		Vec3f center;
		Vec3f halfLength;

		Quat1f rot;
	};

	struct SphereInfo
	{
		SphereInfo()
		{
			center = Vec3f(0.0f, 0.0f, 0.0f);
			radius = 1.0;
			rot = Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
		}
		Quat1f rot;
		Vec3f center;
		float radius;
	};

	struct TetInfo
	{
		TetInfo()
		{
			v[0] = Vec3f(0);
			v[1] = Vec3f(1, 0 ,0);
			v[2] = Vec3f(0, 1, 0);
			v[3] = Vec3f(0, 0, 1);
		}

		Vec3f v[4];
	};

	struct CapsuleInfo
	{
		CapsuleInfo()
		{
			center = Vec3f(0.0f, 0.0f, 0.0f);
			rot = Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
			radius = 1.0f;
			halfLength = 1.0f;
		}
		Quat1f rot;
		Vec3f center;
		float halfLength;
		float radius;
	};

	struct RigidBodyInfo
	{
		RigidBodyInfo()
		{
			linearVelocity = Vec3f(0.0f);
			angularVelocity = Vec3f(0.0f);
			position = Vec3f(0.0f);
			mass = -1.0f;
			inertia = Mat3f(0.0f);
			friction = 0.0f;
			restitution = 0.0f;
			motionType = MT_Static;
			collisionType = CT_AllObjects;
			shapeType = ST_Other;
			angle = Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
		}

		Quat1f angle;

		/// The linear velocity of the body's origin in world co-ordinates.
		Vec3f linearVelocity;

		/// The angular velocity of the body.
		Vec3f angularVelocity;

		/// The world position of the body.
		Vec3f position;

		/// The inertia of the body
		Mat3f inertia;

 		float mass;

		float friction;

		float restitution;

 		MotionType motionType;
		
		ShapeType shapeType;

		CollisionMask collisionType;
	};

	enum ContactType
	{
		CT_BOUDNARY = 0,
		CT_NONPENETRATION,
		CT_FRICTION
	};

// 	struct ContactPair
// 	{
// 		Vec3f pos0;
// 		Vec3f pos1;
// 
// 		Vec3f normal0;
// 		Vec3f normal1;
// 
// 		int id0;
// 		int id1;
// 
// 		ContactType cType;
// 		float distance;
// 	};
}