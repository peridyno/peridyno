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

	struct RigidBodyInfo
	{
		RigidBodyInfo()
		{
			linearVelocity = Vector<Real, 3>(0.0f);
			angularVelocity = Vector<Real, 3>(0.0f);
			position = Vector<Real, 3>(0.0f);
			offset = Vector<Real, 3>(0.0f);
			bodyId = 0;
			mass = -1.0f;
			inertia = SquareMatrix<Real, 3>(0.0f);
			friction = 1.0f;
			restitution = 0.0f;
			motionType = BodyType::Dynamic;
			collisionMask = CT_AllObjects;
			shapeType = ET_Other;
			angle = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f);
		}

		RigidBodyInfo(Vector<Real, 3> p, Quat<Real> q = Quat<Real>(0.0f, 0.0f, 0.0f, 1.0f))
		{
			linearVelocity = Vector<Real, 3>(0.0f);
			angularVelocity = Vector<Real, 3>(0.0f);
			position = p;
			offset = Vector<Real, 3>(0.0f);
			bodyId = 0;
			mass = -1.0f;
			inertia = SquareMatrix<Real, 3>(0.0f);
			friction = 1.0f;
			restitution = 0.0f;
			motionType = BodyType::Dynamic;
			collisionMask = CT_AllObjects;
			shapeType = ET_Other;
			angle = q;
		}

		Quat<Real> angle;

		/// The linear velocity of the body's origin in world co-ordinates.
		Vector<Real, 3> linearVelocity;

		/// The angular velocity of the body.
		Vector<Real, 3> angularVelocity;

		/// The barycenter of the body.
		Vector<Real, 3> position;

		/// An offset from the barycenter to the geometric center
		Vector<Real, 3> offset;

		/// The inertia of the body
		SquareMatrix<Real, 3> inertia;

		uint bodyId;

 		Real mass;

		Real friction;

		Real restitution;

 		BodyType motionType;
		
		ElementType shapeType;

		CollisionMask collisionMask;
	};
}
