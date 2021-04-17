#include "gtest/gtest.h"
#include "Vector.h"

using namespace dyno;

TEST(Vector, cout)
{
	Vec2f vec2(1, 2);
	std::cout << vec2 << std::endl;

	Vec3f vec3(1, 2, 3);
	std::cout << vec3 << std::endl;

	Vec4f vec4(1, 2, 3, 4);
	std::cout << vec4 << std::endl;
}