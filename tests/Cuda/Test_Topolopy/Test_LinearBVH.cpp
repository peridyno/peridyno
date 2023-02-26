#include "gtest/gtest.h"

#include "Topology/LinearBVH.h"

using namespace dyno;

typedef typename TAlignedBox3D<float> AABB;

TEST(BVH, construct)
{
	std::vector<AABB> hAABBs;
	hAABBs.push_back(AABB(Vec3f(0), Vec3f(0.5)));
	hAABBs.push_back(AABB(Vec3f(0.5), Vec3f(1.0)));

	DArray<AABB> dAABBs;
	dAABBs.assign(hAABBs);

	LinearBVH<DataType3f> lbvh;
	lbvh.construct(dAABBs);

	CArray<AABB> hSortedAABBs;
	hSortedAABBs.assign(lbvh.getSortedAABBs());

	AABB root = hSortedAABBs[0];

	EXPECT_EQ((root.length(0) - 1.0f) < EPSILON, true);
}
