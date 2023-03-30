#include "gtest/gtest.h"
#include "VkTransfer.h"
#include <numeric>
using namespace dyno;

TEST(VkTransfer, copy)
{
	std::vector<float> vec(10, 0.0f);
	std::iota(vec.begin(), vec.end(), 1.0f);
	VkDeviceArray<float> d_vec;
	d_vec.resize(10);
	vkTransfer(d_vec, vec);

	std::vector<float> vec2(10);
	vkTransfer(vec2, d_vec);

	for (std::size_t i = 1; i <= 10; i++)
	{
		EXPECT_EQ(vec2[i - 1] == 1.0f*i, true);
	}

	VkDeviceArray<float> d_vec2(2 * d_vec.size());
	vkTransfer(d_vec2, 0, d_vec, 0, d_vec.size());
	vkTransfer(d_vec2, d_vec.size(), d_vec, 0, d_vec.size());

	std::vector<float> h_vec2(d_vec2.size());
	vkTransfer(h_vec2, d_vec2);

	vec.clear();
	d_vec.clear();
	vec2.clear();
	h_vec2.clear();
}