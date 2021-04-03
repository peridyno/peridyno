#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Algorithm/Reduction.h"
using namespace dyno;

TEST(Reduction, accumulate)
{
	CArray<float> cArr;
	cArr.pushBack(1.0f);
	cArr.pushBack(2.0f);
	cArr.pushBack(3.0f);

	DArray<float> dArr;
	dArr.assign(cArr);

	Reduction<float> reduce;
	float accVal = reduce.accumulate(dArr.begin(), dArr.size());

	EXPECT_EQ(abs(accVal - 6.0f) < EPSILON, true);
}