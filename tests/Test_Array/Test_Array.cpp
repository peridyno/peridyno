#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Array/ArrayList.h"
#include "Utility/Function1Pt.h"
#include <thrust/sort.h>
#include "Utility/CTimer.h"
#include "Array/ArrayCopy.h"

#include <list>

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
}

TEST(Array, Copy)
{
	CArray<int> cArr;
	GArray<int> gArr;

	cArr.pushBack(1);
	cArr.pushBack(2);

	gArr.resize(cArr.size());

	arryCpy(gArr, cArr);
}