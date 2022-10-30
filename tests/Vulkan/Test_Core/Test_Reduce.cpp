#include "gtest/gtest.h"
#include <numeric>
#include "VkSystem.h"
#include "VkTransfer.h"
#include "Catalyzer/VkReduce.h"

using namespace px;

TEST(VkReduce, reduce)
{
	VkSystem::instance()->initialize();

	//Test float type Data
	VkReduce<float> sortFloat;
	std::vector<float> inputFloatData(100, 1.0f);
	std::iota(inputFloatData.begin(), inputFloatData.end(), 1.0f);
	EXPECT_EQ(sortFloat.reduce(inputFloatData) == 5050.0f, true);

	//Test int type Data
	VkReduce<int> sortInt;
	std::vector<int> inputIntData2(100, 1);
	std::iota(inputIntData2.begin(), inputIntData2.end(), 1);
	EXPECT_EQ(sortInt.reduce(inputIntData2) == 5050, true);


	//Test uint type Data
	VkReduce<uint32_t> sortUint;
	std::vector<uint32_t> inputIntData3(100, 1);
	std::iota(inputIntData3.begin(), inputIntData3.end(), 1);
	EXPECT_EQ(sortUint.reduce(inputIntData3) == 5050, true);

}