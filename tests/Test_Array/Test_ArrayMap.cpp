#include "gtest/gtest.h"
#include "STL/Map.h"
#include "STL/Pair.h"
#include <vector>

using namespace dyno;

TEST(Map, CPU)
{
	Pair<int, double> pair1(1, 1.1), pair2(2, 2.2), pair3(3, 3.3), pair4(4, 4.4);

	Map<int, double> mymap;
	if (mymap.empty()) std::cout << "this map is empty" << std::endl;

	Pair<int, double>* pairptr = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
	mymap.reserve(pairptr, 4);
	if (mymap.empty()) std::cout << "this map is empty" << std::endl;

	Pair<int, double>* pairp=mymap.insert(pair1);
	EXPECT_EQ(pairp->first, 1);
	EXPECT_EQ(pairp->second, 1.1);

	pairp = mymap.insert(pair4);
	EXPECT_EQ(pairp->first, 4);
	EXPECT_EQ(pairp->second, 4.4);

	pairp = mymap.insert(pair2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 2.2);

	pairp = mymap.insert(pair2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 4.4);

	pairp = mymap.begin();
	EXPECT_EQ(pairp->first, 1);
	EXPECT_EQ(pairp->second, 1.1);
	EXPECT_EQ((pairp+1)->first, 2);
	EXPECT_EQ((pairp+1)->second, 4.4);
	EXPECT_EQ((pairp+2)->first, 4);
	EXPECT_EQ((pairp+2)->second, 4.4);

	pairp = mymap.find(2);
	EXPECT_EQ(pairp->first, 2);
	EXPECT_EQ(pairp->second, 4.4);
	pairp = mymap.find(3);
	EXPECT_EQ(pairp, nullptr);

	free(pairptr);
}

//TEST(ArrayList, Copy)
//{
//	std::vector<std::vector<int>> vvec;
//	std::vector<int> vec1;
//	std::vector<int> vec2;
//
//	vec1.push_back(1);
//	vec1.push_back(2);
//
//	vec2.push_back(2);
//	vvec.push_back(vec1);
//	vvec.push_back(vec2);
//
//	DArrayList<int> gArrList;
//	gArrList.assign(vvec);
//
//	CArrayList<int> cArrList;
//	cArrList.assign(gArrList);
//
//	std::cout << "Device ArrayList: " << cArrList;
//	std::cout << "Host ArrayList: " << gArrList;
//
//	auto iter = cArrList[0].begin();
//	EXPECT_EQ(*iter, 1);
//
//	iter++;
//	EXPECT_EQ(*iter, 2);
//}


