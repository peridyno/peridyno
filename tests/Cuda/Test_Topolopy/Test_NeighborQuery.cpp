#include "gtest/gtest.h"
#include "Topology/NeighborPointQuery.h"

using namespace dyno;

TEST(NeighborPointQuery, findNeighbors1D)
{
	NeighborPointQuery<DataType3f> nQuery;

	CArray<Vec3f> points;

	for (float x = 0.0f; x < 1.0f; x += 0.1f)
	{
		points.pushBack(Vec3f(x, 0.0f, 0.0f));
	}

	nQuery.inRadius()->setValue(1.01f);
	nQuery.inPosition()->allocate()->assign(points);
	nQuery.update();

	auto& nbrIds = nQuery.outNeighborIds()->getData();

	std::cout << nbrIds.elements() << std::endl;
	std::cout << nbrIds << std::endl;
}

TEST(NeighborPointQuery, findNeighbors)
{
	NeighborPointQuery<DataType3f> nQuery;

	CArray<Vec3f> points;

	for (float x = 0.0f; x < 1.0f; x += 0.1f)
	{
		for (float y = 0.0f; y < 1.0f; y += 0.1f)
		{
			for (float z = 0.0f; z < 1.0f; z += 0.1f)
			{
				points.pushBack(Vec3f(x, y, z));
			}
		}
	}

	nQuery.inRadius()->setValue(0.12f);
	nQuery.inPosition()->allocate()->assign(points);
	nQuery.update();

	auto& nbrIds = nQuery.outNeighborIds()->getData();

	CArrayList<int> host_nbrIds;
	host_nbrIds.assign(nbrIds);

	EXPECT_EQ(host_nbrIds[0].size() == 4, true);

	CArray<Vec3f> points2;
	points2.pushBack(Vec3f(0.2f));
	nQuery.inOther()->allocate()->assign(points2);
	nQuery.update();

	host_nbrIds.assign(nbrIds);
	EXPECT_EQ(host_nbrIds[0].size() == 7, true);

	nQuery.varSizeLimit()->setValue(2);
	nQuery.update();

	host_nbrIds.assign(nbrIds);
	EXPECT_EQ(host_nbrIds[0].size() == 2, true);
}
