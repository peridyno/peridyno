#include "gtest/gtest.h"
#include "Algorithm/ForEach.h"

using namespace dyno;

TEST(Lambdas, globalFunc)
{
	size_t size = 10;
	float* vector = new float[size];
	float a = 1.0f;

	auto assign = [=]  __host__ __device__(size_t i) { vector[i] = a; };

	ForEach(30);
}
