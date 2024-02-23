#include "gtest/gtest.h"
#include "VkSystem.h"
#include "VkTransfer.h"
#include "Catalyzer/VkSort.h"

using namespace dyno;

TEST(VkSort, sort)
{
	VkSystem::instance()->initialize();
	VkSortByKey<float, float> sortInt;
	uint32_t dSize = 10;
	CArray<float> keys;
	keys.assign(std::vector<float>(dSize, 1.0f));
	CArray<float> values;
	values.assign(std::vector<float>(dSize, 1.0f));

	for (std::size_t i = 0; i < dSize; i++)
	{
		keys[i] = rand() % 2048;
		values[i] = rand() % 2048;
		//printf("keys[%d]=%d  values[%d]=%d \n", i, keys[i], i, values[i]);
	}
	//test sort by key
	//sortInt.sort_by_key(keys, values, UP);
	sortInt.sortByKey(keys, values, SortParam::eUp);
	int SortType = SortParam::eUp;

	//test sort
	//sortInt.sort(keys,DOWN);
	//int SortType = DOWN;

	for (std::size_t i = 0; i < dSize - 1; i++)
	{
		if (SortType == 0)
			EXPECT_EQ(keys[i] <= keys[i + 1], true);
		else
			EXPECT_EQ(keys[i] >= keys[i + 1], true);
		//printf("key[%d]=%d  values[%d]=%d\n", i, keys[i], i, values[i]);
	}
}