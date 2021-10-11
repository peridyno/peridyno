#include "gtest/gtest.h"

#include "Quat.h"
#include "Topology/Primitive3D.h"
#include "Collision/CollisionDetectionAlgorithm.h"

using namespace dyno;


TEST(TET, collision)
{
	using Tet = TTet3D<float>;

	Tet t0;
	Tet t1;


	//one contact point
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0, 0.96, 0), Coord3D(1, 0.96, 0), Coord3D(0, 1.96, 0), Coord3D(0, 0.96, 1));


	TManifold<Real> manifold;
	CollisionDetection<float>::request(manifold, t0, t1);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.04f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[0] - 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1] - 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2] - 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);

	//test reverse axis
	CollisionDetection<float>::request(manifold, t1, t0);
	EXPECT_EQ(manifold.contactCount == 1, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.04f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1] + 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2] + 1.0f / std::sqrt(3.0f)) < REAL_EPSILON, true);
	
	
	//edge-edge
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.4, -0.5, 0.4), Coord3D(1.4, 0.46, 0.4), 
		Coord3D(0.4, 0.46, 0.4), Coord3D(0.4, 0.46, 1.4));
	CollisionDetection<float>::request(manifold, t1, t0);
	EXPECT_EQ(manifold.contactCount == 1, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f * std::sqrt(2.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f / std::sqrt(2.0f)) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2] + 1.0f / std::sqrt(2.0f)) < REAL_EPSILON, true);


	//edge-face, one inside
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.1, 0.2, 0.1), Coord3D(0.1, 1.46, 0.1),
		Coord3D(-0.4, 0.2, 0.2), Coord3D(-0.4, 0.2, -0.5));
	CollisionDetection<float>::request(manifold, t0, t1);
	EXPECT_EQ(manifold.contactCount == 2, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2]) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[1] - 0.2f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[2] - 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[1].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].position[1] - 0.9f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].position[2] - 0.1f) < REAL_EPSILON, true);

	//edge-face, two inside
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.1, 0.2, 0.1), Coord3D(0.1, 0.6, 0.1),
		Coord3D(-0.4, 0.2, 0.2), Coord3D(-0.4, 0.2, -0.5));
	CollisionDetection<float>::request(manifold, t0, t1);
	EXPECT_EQ(manifold.contactCount == 2, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2]) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[1] - 0.2f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[2] - 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[1].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].position[1] - 0.6f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].position[2] - 0.1f) < REAL_EPSILON, true);

	
	
	//edge-face, no inside
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.1, -0.2, 0.1), Coord3D(0.1, 1.6, 0.1),
		Coord3D(-0.4, -0.2, 0.2), Coord3D(-0.4, -0.2, -0.5));
	CollisionDetection<float>::request(manifold, t0, t1);
	EXPECT_EQ(manifold.contactCount == 2, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2]) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[0].position[2] - 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.contacts[1].position[0] - 0.1f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(std::abs(manifold.contacts[1].position[1] - manifold.contacts[0].position[1]) - 0.9f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.contacts[1].position[2] - 0.1f) < REAL_EPSILON, true);



	//face-face, no inside
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.1, 0.6, 0.6), Coord3D(0.1, -0.6, 0.6),
		Coord3D(-0.9, 0.6, 0.6), Coord3D(0.1, 0.6, -0.4));
	CollisionDetection<float>::request(manifold, t0, t1);
	EXPECT_EQ(manifold.contactCount == 6, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.normal[0] + 1.0f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2]) < REAL_EPSILON, true);

	
	//face-face, with inside
	t0 = Tet(Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	t1 = Tet(Coord3D(0.1, 0.6, 0.6), Coord3D(0.1, -0.6, 0.6),
		Coord3D(-0.9, 0.6, 0.6), Coord3D(0.1, 0.6, -0.9));
	CollisionDetection<float>::request(manifold, t1, t0);
	EXPECT_EQ(manifold.contactCount == 5, true);

	EXPECT_EQ(std::abs(manifold.contacts[0].penetration + 0.1f) < REAL_EPSILON, true);

	EXPECT_EQ(std::abs(manifold.normal[0] - 1.0f) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[1]) < REAL_EPSILON, true);
	EXPECT_EQ(std::abs(manifold.normal[2]) < REAL_EPSILON, true);
	


	/*printf("%d contact normal: %.3lf %.3lf %.3lf\n", manifold.contactCount, manifold.normal[0], manifold.normal[1], manifold.normal[2]);
	printf("contact point 1: %.3lf %.3lf %.3lf\n", 
		manifold.contacts[0].position[0],
		manifold.contacts[0].position[1],
		manifold.contacts[0].position[2]);
	printf("contact point 2: %.3lf %.3lf %.3lf\n",
		manifold.contacts[1].position[0],
		manifold.contacts[1].position[1],
		manifold.contacts[1].position[2]);
	printf("contact point 2: %.3lf %.3lf %.3lf\n",
		manifold.contacts[2].position[0],
		manifold.contacts[2].position[1],
		manifold.contacts[2].position[2]);
	printf("contact point 2: %.3lf %.3lf %.3lf\n",
		manifold.contacts[3].position[0],
		manifold.contacts[3].position[1],
		manifold.contacts[3].position[2]);
	printf("contact penetration: %.3lf\n",
		manifold.contacts[0].penetration
		);*/
 	
}