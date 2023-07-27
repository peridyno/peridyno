#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

#include "Collision/CollisionData.h"
#include "Topology/DiscreteElements.h"

namespace dyno
{
	enum MotionType
	{
		MT_Static = 0,
		MT_Kinematic,
		MT_Dynamic,
	};

	struct BoxInfo
	{
		BoxInfo()
		{
			center = Vector<Real, 3>(0.0f, 0.0f, 0.0f);
			halfLength = Vector<Real, 3>(1.0f, 1.0f, 1.0f);
			rot = Quat<Real>(1.0f, 0.0f, 0.0f, 0.0f);
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
			rot = Quat<Real>(1.0f, 0.0f, 0.0f, 0.0f);
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
			v[1] = Vec3f(1, 0 ,0);
			v[2] = Vec3f(0, 1, 0);
			v[3] = Vec3f(0, 0, 1);
		}

		Vector<Real, 3> v[4];
	};
	// Custom get and set functions for v member
	static void set_v(TetInfo& obj, const std::vector<Vec3f>& values) {
		if (values.size() != 4) {
			throw std::runtime_error("TetInfo.v must be a list of 4 Vec3f elements.");
		}
		for (size_t i = 0; i < 4; ++i) {
			obj.v[i] = values[i];
		}
	}

	static std::vector<Vec3f> get_v(const TetInfo& obj) {
		std::vector<Vec3f> values;
		for (size_t i = 0; i < 4; ++i) {
			values.push_back(obj.v[i]);
		}
		return values;
	}

	struct CapsuleInfo
	{
		CapsuleInfo()
		{
			center = Vec3f(0.0f, 0.0f, 0.0f);
			rot = Quat1f(1.0f, 0.0f, 0.0f, 0.0f);
			radius = 1.0f;
			halfLength = 1.0f;
		}
		Quat<Real> rot;
		Vector<Real, 3> center;
		Real halfLength;
		Real radius;
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
			collisionMask = CT_AllObjects;
			shapeType = ET_Other;
			angle = Quat1f(0.0f, 0.0f, 0.0f, 1.0f);
		}

		Quat<Real> angle;

		/// The linear velocity of the body's origin in world co-ordinates.
		Vector<Real, 3> linearVelocity;

		/// The angular velocity of the body.
		Vector<Real, 3> angularVelocity;

		/// The world position of the body.
		Vector<Real, 3> position;

		/// The inertia of the body
		SquareMatrix<Real, 3> inertia;

 		Real mass;

		Real friction;

		Real restitution;

 		MotionType motionType;
		
		ElementType shapeType;

		CollisionMask collisionMask;
	};
}
