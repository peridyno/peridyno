#include "gtest/gtest.h"

#include "CCD/TightCCD.h"

#include "Topology/Primitive3D.h"

using namespace dyno;

TEST(TightCCD, VertexFaceCCD)
{
    const Vec3f a0s(0.0, 0.0, 0.0);
    const Vec3f a1s(1, 0, 0);
    const Vec3f a0e(0, 1, 0);
    const Vec3f a1e(0, 0, 1);
    const Vec3f b0s(1.0, 1.0, 1.0);
    const Vec3f b1s(0.5, 0, 0);
    const Vec3f b0e(0, 0.5, 0);
    const Vec3f b1e(0, 0, 0.5);

    float toi = 1.0f;
    bool res = TightCCD<float>::VertexFaceCCD(
        a0s, a1s, a0e, a1e, b0s, b1s, b0e, b1e, toi);

    EXPECT_EQ(res, true);
    EXPECT_EQ(std::abs(toi - 0.285714298) < REAL_EPSILON, true);
}

TEST(TightCCD, EdgeEdgeCCD)
{
	const Vec3f a0s(0.0, 0.0, 0.0);
	const Vec3f a1s(1, 1, 0);
	const Vec3f a0e(1, 0, 1);
	const Vec3f a1e(0, 1, 1);
	const Vec3f b0s(0.0, 0.0, 1.0);
	const Vec3f b1s(1.0, 1.0, 1.0);
	const Vec3f b0e(1, 0, 0);
	const Vec3f b1e(0, 1, 0);

	float toi = 1.0f;
    bool res = TightCCD<float>::EdgeEdgeCCD(
		a0s, a1s, a0e, a1e, b0s, b1s, b0e, b1e, toi);

	EXPECT_EQ(res, true);
	EXPECT_EQ(std::abs(toi - 0.5) < REAL_EPSILON, true);
}

TEST(TightCCD, Triangle)
{
    Triangle3D s0(Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(0, 1, 0));
    Triangle3D s1(Vec3f(0, 0, 1), Vec3f(1, 0, 1), Vec3f(0, 1, 1));

	Triangle3D t0(Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(0, 1, 0));
	Triangle3D t1(Vec3f(0, 0, -1), Vec3f(1, 0, -1), Vec3f(0, 1, -1));

	float toi = 1.0f;
	bool res = TightCCD<float>::TriangleCCD(
		s0, s1, t0, t1, toi);

	EXPECT_EQ(res, true);
	EXPECT_EQ(std::abs(toi) < REAL_EPSILON, true);
}