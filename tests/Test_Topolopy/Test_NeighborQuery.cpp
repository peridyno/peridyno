#include "gtest/gtest.h"
#include "Topology/NeighborPointQuery.h"

using namespace dyno;

TEST(NeighborPointQuery, findNeighbors)
{
	NeighborPointQuery<DataType3f> nQuery;

	CArray<Vector3f> points;

	for (float x = 0.0f; x < 1.0f; x += 0.1f)
	{
		for (float y = 0.0f; y < 1.0f; y += 0.1f)
		{
			for (float z = 0.0f; z < 1.0f; z += 0.1f)
			{
				points.pushBack(Vector3f(x, y, z));
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

	CArray<Vector3f> points2;
	points2.pushBack(Vector3f(0.2f));
	nQuery.inOther()->allocate()->assign(points2);
	nQuery.update();

	host_nbrIds.assign(nbrIds);
	EXPECT_EQ(host_nbrIds[0].size() == 7, true);

	nQuery.varSizeLimit()->setValue(2);
	nQuery.update();

	host_nbrIds.assign(nbrIds);
	EXPECT_EQ(host_nbrIds[0].size() == 2, true);
}
