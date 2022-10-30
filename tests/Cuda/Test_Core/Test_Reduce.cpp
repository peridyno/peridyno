#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Algorithm/Reduction.h"

#include "Timer.h"
using namespace dyno;

TEST(Reduction, accumulate)
{
	GTimer timer;

	CArray<float> cArr;
	cArr.pushBack(1.0f);
	cArr.pushBack(2.0f);
	cArr.pushBack(3.0f);

	DArray<float> dArr;
	dArr.assign(cArr);

	timer.start();
	Reduction<float> reduce;
	float accVal = reduce.accumulate(dArr.begin(), dArr.size());
	float minVal = reduce.minimum(dArr.begin(), dArr.size());
	timer.stop();
	std::cout << "Reduce Time: " << timer.getEclipsedTime() << std::endl;


	EXPECT_EQ(abs(accVal - 6.0f) < EPSILON, true);
}