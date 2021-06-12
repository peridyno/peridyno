#include "gtest/gtest.h"
#include "Volume/VolumeGenerator.h"

using namespace dyno;

TEST(Volume, generator)
{
	auto sdfGen = std::make_shared<VolumeGenerator<DataType3f>>();
	sdfGen->load("../../data/standard/standard_cube.obj");
}
