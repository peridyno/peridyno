#include "gtest/gtest.h"

#include "CCD/TightCCD.h"

#include "Topology/Primitive3D.h"

using namespace dyno;

TEST(CCD, VertexFaceCCD)
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
    bool res = VertexFaceCCD(
        a0s, a1s, a0e, a1e, b0s, b1s, b0e, b1e, toi);

    EXPECT_EQ(res, true);
    EXPECT_EQ(std::abs(toi - 0.285714298) < REAL_EPSILON, true);
}

TEST(CCD, EdgeEdgeCCD)
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
    bool res = EdgeEdgeCCD(
		a0s, a1s, a0e, a1e, b0s, b1s, b0e, b1e, toi);

	EXPECT_EQ(res, true);
	EXPECT_EQ(std::abs(toi - 0.5) < REAL_EPSILON, true);
}