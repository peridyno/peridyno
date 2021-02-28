#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Array/ArrayList.h"
#include "Utility/Function1Pt.h"
#include <thrust/sort.h>

using namespace dyno;

TEST(Array, CPU)
{
	CArray<int> cArr;

	cArr.push_back(1);
	cArr.push_back(2);

	EXPECT_EQ(cArr.size(), 2);

	GArray<int> gArr;
	gArr.resize(2);
	
	Function1Pt::copy(gArr, cArr);

	GArrayList<int> arrList;
	arrList.resize(gArr);

	EXPECT_EQ(arrList.elementSize(), 3);
}