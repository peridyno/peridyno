#include "gtest/gtest.h"
#include "STL/Map.h"
#include "STL/Pair.h"
#include "Array/ArrayMap.h"
#include <vector>

using namespace dyno;

TEST(Map, CPU)
{
	Pair<int, double> pair1(1, 1.1), pair2(2, 2.2), pair3(3, 3.3), pair4(4, 4.4);

	Map<int, double> mymap1,mymap2,mymap3;
	if (mymap1.empty()) std::cout << "this map is empty" << std::endl;

	Pair<int, double>* pairptr1 = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
	Pair<int, double>* pairptr2 = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
	mymap1.reserve(pairptr1, 4);
	mymap2.reserve(pairptr2, 4);

	Pair<int, double>* pairp=mymap1.insert(pair1);
	EXPECT_EQ(pairp->first, 1);
	EXPECT_EQ(pairp->second, 1.1);

	pairp = mymap1.insert(pair4);
	EXPECT_EQ(pairp->first, 4);
	EXPECT_EQ(pairp->second, 4.4);

	pairp = mymap1.insert(pair2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 2.2);

	pairp = mymap1.insert(pair2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 4.4);

	pairp = mymap1.begin();
	EXPECT_EQ(pairp->first, 1);
	EXPECT_EQ(pairp->second, 1.1);
	EXPECT_EQ((pairp+1)->first, 2);
	EXPECT_EQ((pairp+1)->second, 4.4);
	EXPECT_EQ((pairp+2)->first, 4);
	EXPECT_EQ((pairp+2)->second, 4.4);

	pairp = mymap1.find(2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 4.4);
	pairp = mymap1.find(3);
	EXPECT_EQ(pairp, nullptr);

	pairp = mymap2.insert(pair2);
	pairp = mymap2.insert(pair3);
	pairp = mymap2.insert(pair4);
	pairp = mymap2.insert(pair1);

	std::vector<Map<int, double>> mapvec;
	mapvec.push_back(mymap1);
	mapvec.push_back(mymap3);
	mapvec.push_back(mymap2);

	DArrayMap<double> dArrMap;
	dArrMap.assign(mapvec);

	CArrayMap<double> cArrMap;
	cArrMap.assign(dArrMap);

	std::cout << "Host ArrayMap: " << cArrMap;
	std::cout << "Device ArrayMap: " << dArrMap;


	EXPECT_EQ(cArrMap[0].size(), 3);
	EXPECT_EQ(cArrMap[1].size(), 0);
	EXPECT_EQ(cArrMap[2].size(), 4);

	free(pairptr1);
	free(pairptr2);
}




