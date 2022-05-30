#include "gtest/gtest.h"
#include "Topology/SparseOctree.h"

using namespace dyno;

TEST(Octree, MortonCode)
{
	OctreeNode c1(4, 15, 7, 3);
	
	EXPECT_EQ(c1.key(), 29439);

	OctreeNode c2(4, 3, 1, 0);

	EXPECT_EQ(c2.key(), 28683);

	EXPECT_EQ(c1.leastCommonAncestor(c2), 7);

	OctreeNode c3(3, 1, 1, 1);
	OctreeNode c4(4, 1, 1, 1);

	auto k3 = c3.key();
	auto k4 = c4.key();
	EXPECT_EQ(c3.leastCommonAncestor(c4), 448);


	OctreeNode c0000(0, 0, 0, 0);
	OctreeNode c1000(1, 0, 0, 0);

	EXPECT_EQ(c0000.leastCommonAncestor(c1000).key() == c0000.key(), true);
}


TEST(Octree, Compare)
{
	OctreeNode c1(4, 0, 0, 0);
	OctreeNode c2(3, 4, 4, 4);

	EXPECT_EQ(c2 > c1, true);

	OctreeNode c3(3, 0, 0, 0);

	EXPECT_EQ(c2 > c3, true);

	OctreeNode c4(4, 4, 4, 4);

	EXPECT_EQ(c1 < c4, true);
	EXPECT_EQ(c1 <= c4, true);
	EXPECT_EQ(c4 >= c1, true);
	EXPECT_EQ(c4 > c1, true);
	EXPECT_EQ(c4 == c1, false);

	OctreeNode c5(2, 0, 0, 0);
	OctreeNode c6(3, 0, 0, 0);

	EXPECT_EQ(c6 > c5, true);
	EXPECT_EQ(c5 > c6, false);
	EXPECT_EQ(c5 >= c6, false);
	EXPECT_EQ(c6 >= c5, true);
	EXPECT_EQ(c6 > c5, true);
	EXPECT_EQ(c6 == c5, false);

	EXPECT_EQ(c2 > c5, true);
	EXPECT_EQ(c2 >= c5, true);
}

TEST(Octree, Construct)
{
	CArray<Vec3f> h_arr;
	DArray<Vec3f> d_arr;
	h_arr.resize(4);
	d_arr.resize(4);
	for (int i = 0; i < 4; i++)
	{
		h_arr[i] = Vec3f(i) + 0.5f;
	}
	h_arr[0] = Vec3f(1.5f);
	h_arr[3] = Vec3f(3.0f);
	d_arr.assign(h_arr);

	SparseOctree<DataType3f> octree;
	octree.setSpace(Vec3f(0), 1, 4);
	octree.construct(d_arr, 0.2);

	OctreeNode node = octree.queryNode(2, 3, 3, 3);

	octree.printPostOrderedTree();

	printf("m_data_size = %d\n", node.m_data_size);

	EXPECT_EQ(node.m_data_size == 1, true);

}
