#include "gtest/gtest.h"
#include "VkSystem.h"
#include "VkTransfer.h"
#include "Catalyzer/VkSort.h"

using namespace dyno;

TEST(VkSort, sort)
{
	VkSortByKey<int, int> sortInt;
	uint32_t dSize = 10;
	CArray<int> keys;
	keys.assign(std::vector<int>(dSize, 1.0f));
	CArray<int> values;
	values.assign(std::vector<int>(dSize, 1.0f));

	for (std::size_t i = 0; i < dSize; i++)
	{
		keys[i] = rand() % 2048;
		values[i] = rand() % 2048;
		//printf("keys[%d]=%d  values[%d]=%d \n", i, keys[i], i, values[i]);
	}

	sortInt.sortByKey(keys, values, SortParam::eUp);

	for (std::size_t i = 0; i < dSize - 1; i++)
	{
		EXPECT_EQ(keys[i] <= keys[i + 1], true);
	}

	sortInt.sortByKey(keys, values, SortParam::eDown);
	for (std::size_t i = 0; i < dSize - 1; i++)
	{
		EXPECT_EQ(keys[i] >= keys[i + 1], true);
	}
}