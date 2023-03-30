#include "gtest/gtest.h"
#include "Volume/VolumeGenerator.h"

using namespace dyno;

TEST(Volume, generator)
{
	auto sdfGen = std::make_shared<VolumeGenerator<DataType3f>>();
	sdfGen->load("../../data/standard/standard_cube.obj");

//	float val = sdfGen->phi(50, 50, 50);

	for (uint i = 0; i < sdfGen->phi.size(); i++)
	{
		if (sdfGen->phi[i] < -6.0 || sdfGen->phi[i] > 6.0)
		{
			printf("%u: %f \n", i, sdfGen->phi[i]);
		}
	}
}
