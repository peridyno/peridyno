#include "gtest/gtest.h"

#include "DirectedAcyclicGraph.h"
using namespace dyno;

TEST(DAG, topologicalSortFromNode)
{
	// Create a graph given in the above diagram
	DirectedAcyclicGraph g;
	g.addEdge(0, 1);
	g.addEdge(0, 9);
	g.addEdge(1, 2);
	g.addEdge(2, 3);
	g.addEdge(9, 3);
	g.addEdge(10, 9);

	auto& list2 = g.topologicalSort(2);
	EXPECT_EQ(list2.size(), 2);
	EXPECT_EQ(list2[0], 2);
	EXPECT_EQ(list2[1], 3);

	auto& list = g.topologicalSort();
	EXPECT_EQ(list.size(), 6);
	EXPECT_EQ(list[5], 3);
}