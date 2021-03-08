#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Array/ArrayList.h"
#include <thrust/sort.h>
#include <vector>

using namespace dyno;

TEST(Array, CPU)
{
	CArray<int> cArr;

	cArr.pushBack(1);
	cArr.pushBack(2);

	EXPECT_EQ(cArr.size(), 2);

	GArray<int> gArr;
	gArr.resize(2);
	
	Function1Pt::copy(gArr, cArr);

	GArrayList<int> arrList;
	arrList.resize(gArr);

	EXPECT_EQ(arrList.elementSize(), 3);

	GArrayList<int> constList;
	constList.resize(5, 2);

	EXPECT_EQ(constList.elementSize(), 10);

	std::cout << cArr;
	std::cout << gArr;
}

TEST(Array, Copy)
{
	CArray<int> cArr;
	CArray<int> cArr2;
	GArray<int> gArr;
	GArray<int> gArr2;

	cArr.pushBack(1);
	cArr.pushBack(2);

	gArr.resize(cArr.size());

	cArr2.assign(cArr);
	gArr.assign(cArr);

	cArr.assign(gArr);

	gArr2.resize(gArr.size());
//	gArr2.copyFrom(gArr);
}

TEST(ArrayList, Copy)
{
	std::vector<std::vector<int>> vvec;
	std::vector<int> vec1;
	std::vector<int> vec2;

	vec1.push_back(1);
	vec1.push_back(2);

	vec2.push_back(2);
	vvec.push_back(vec1);
	vvec.push_back(vec2);

	GArrayList<int> gArrList;
	gArrList.assign(vvec);

	CArrayList<int> cArrList;
	cArrList.assign(gArrList);

	std::cout << "Device ArrayList: " << cArrList;
	std::cout << "Host ArrayList: " << gArrList;

	auto iter = cArrList[0].begin();
	EXPECT_EQ(*iter, 1);

	iter++;
	EXPECT_EQ(*iter, 2);
}

TEST(Array2D, CPU)
{

}