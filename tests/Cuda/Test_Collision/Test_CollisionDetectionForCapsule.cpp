#include "gtest/gtest.h"

#include "Quat.h"
#include "Primitive/Primitive3D.h"
#include "Collision/CollisionDetectionAlgorithm.h"

using namespace dyno;

using Coord3D = Vector<Real, 3>;

TEST(Capsule, sphere)
{
	using Sphere = TSphere3D<float>;
	using Capsule = TCapsule3D<float>;

	Sphere sphere = Sphere(Coord3D(0, 0, 0), 1.0f);
	Capsule cap = Capsule(Coord3D(0, 1.99, 0), Quat1f(), 1, 1);

	TManifold<Real> manifold;
	CollisionDetection<float>::request(manifold, sphere, cap);

	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 1.01f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.x) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.y + 0.01f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.z) < REAL_EPSILON, true);

	CollisionDetection<float>::request(manifold, cap, sphere);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 1.01f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.x) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.y - 1.0f) < 10 * REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position.z) < REAL_EPSILON, true);
}

TEST(Capsule, obb)
{
	using Box = TOrientedBox3D<float>;
	using Capsule = TCapsule3D<float>;

	Box b0 = Box(Coord3D(0, 0, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	Capsule cap = Capsule(Coord3D(0, 1.9, 0), Quat1f(), 1, 1);

	TManifold<Real> manifold;
	CollisionDetection<float>::request(manifold, cap, b0);

	EXPECT_EQ(manifold.contactCount == 1, true);

	Capsule cap1 = Capsule(Coord3D(1.9, 0, 0), Quat1f(), 1, 1);
	CollisionDetection<float>::request(manifold, cap1, b0);

	EXPECT_EQ(manifold.contactCount == 2, true);

	Capsule cap2 = Capsule(Coord3D(-1.9, 0, 0), Quat1f(), 1, 1);
	CollisionDetection<float>::request(manifold, cap2, b0);

	EXPECT_EQ(manifold.contactCount == 2, true);

	Capsule cap3 = Capsule(Coord3D(0, 0, 1.9), Quat1f(), 1, 1);
	CollisionDetection<float>::request(manifold, cap3, b0);

	EXPECT_EQ(manifold.contactCount == 2, true);

	Capsule cap4 = Capsule(Coord3D(0, 0, -1.9), Quat1f(), 1, 1);
	CollisionDetection<float>::request(manifold, cap4, b0);

	EXPECT_EQ(manifold.contactCount == 2, true);

	return;
}