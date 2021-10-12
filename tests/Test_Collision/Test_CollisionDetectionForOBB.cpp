#include "gtest/gtest.h"

#include "Quat.h"
#include "Topology/Primitive3D.h"
#include "Collision/CollisionDetectionAlgorithm.h"

using namespace dyno;

TEST(Sphere, collision)
{
	using Box = TOrientedBox3D<float>;
	using Sphere = TSphere3D<float>;

	Box box;
	Sphere sphere;

	box = Box(Coord3D(0, 0, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	sphere = Sphere(Coord3D(0, 2, 0), 1.1f);

	TManifold<Real> manifold;
	CollisionDetection<float>::request(manifold, sphere, box);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	sphere = Sphere(Coord3D(0, 2, 0), 0.9f);
	CollisionDetection<float>::request(manifold, sphere, box);
	EXPECT_EQ(manifold.contactCount == 0, true);
}

TEST(OBB, collision)
{
	using Box = TOrientedBox3D<float>;

	Box b0;
	Box b1;

	b0 = Box(Coord3D(0, 0, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	b1 = Box(Coord3D(0, 1.5, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));

 	TManifold<Real> manifold;
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 4, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[2].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[3].penetration + 0.5f) < REAL_EPSILON, true);

	b1 = Box(Coord3D(1.5, 0, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 4, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[2].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[3].penetration + 0.5f) < REAL_EPSILON, true);

	b1 = Box(Coord3D(0, 0, 1.5), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 4, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[2].penetration + 0.5f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[3].penetration + 0.5f) < REAL_EPSILON, true);

	b0 = Box(Coord3D(0, 0, 0), Quat1f(0.1f, Vec3f(0.3, 0.2, 0.1)), Coord3D(1, 1, 1));
	b1 = Box(Coord3D(0, 1.5, 0), Quat1f(0.2f, Vec3f(0.2, 0.5, 1)), Coord3D(1, 1, 1));

	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.672051370) < REAL_EPSILON, true);

	b0 = Box(Coord3D(0, 0, 0), Quat<float>(0, 0, 0, 1), Coord3D(1, 1, 1));
	b1 = Box(Coord3D(0, 1.5, 0), Quat1f(0.2f, Vec3f(0.2, 0.5, 1)), Coord3D(1, 1, 1));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.708195090) < REAL_EPSILON, true);

	b1 = Box(Coord3D(1.5, 0, 0), Quat1f(0.2f, Vec3f(0.2, 0.5, 1)), Coord3D(1, 1, 1));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.786478937) < REAL_EPSILON, true);

	b1 = Box(Coord3D(0, 0, 1.5), Quat1f(0.2f, Vec3f(0.2, 0.5, 1)), Coord3D(1, 1, 1));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.629602551) < REAL_EPSILON, true);

	b0 = Box(Coord3D(0, 0.0, 0), Quat<float>(1, 0, 0, 0), Coord3D(0.2, 0.2, 0.2));
	b1 = Box(Coord3D(0.000000, 0.415704, 0.000000), Quat<float>(0.000000, 0.000000, 0.049979, 0.998750), Coord3D(0.2, 0.2, 0.2));
	CollisionDetection<float>::request(manifold, b0, b1);
	EXPECT_EQ(manifold.contactCount == 2, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.00326342881) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[0] + 0.179034233) < REAL_EPSILON, true);
}