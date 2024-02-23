
#include "gtest/gtest.h"
#include <numeric>
#include "VkTransfer.h"
#include "Catalyzer/VkScan.h"

using namespace dyno;

TEST(VkScan, scan)
{
	//Test int type Data
	VkScan<int> scanInt;
	std::vector<int> inputData(1000, 1);
	scanInt.scan(inputData , INCLUSIVESCAN);
	int res = inputData.back();
	EXPECT_EQ(res == 1000, true);
	
	//Test float type Data
	VkScan<float> scanFloat;
	std::vector<float> inputData2(1000, 1.0f);
	scanFloat.scan(inputData2 , INCLUSIVESCAN);
	float res2 = inputData2.back();
	EXPECT_EQ(res2 == 1000.0f, true);

	//Test uint type Data
	VkScan<uint32_t> scanUint;
	std::vector<uint32_t> inputData3(1000, 1);
	scanUint.scan(inputData3 , INCLUSIVESCAN);
	uint32_t res3 = inputData3.back();
	EXPECT_EQ(res3 == 1000, true);
	
	
	VkScan<int> scanIntEx;
	std::vector<int> inputData4(10000, 1);
	scanIntEx.scan(inputData4 , EXCLUSIVESCAN);
	uint32_t res4 = inputData4.back();
	EXPECT_EQ(res4 == 9999, true);


}
