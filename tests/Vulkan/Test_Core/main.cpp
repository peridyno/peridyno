#include "gtest/gtest.h"
#include "VkSystem.h"

int main(int argc, char **argv) {
	dyno::VkSystem::instance()->setAssetPath(getAssetPath());
	dyno::VkSystem::instance()->initialize(false);
	
    ::testing::InitGoogleTest(&argc, argv);
    int res =  RUN_ALL_TESTS();

	dyno::VkSystem::destroy();

	return res;
}