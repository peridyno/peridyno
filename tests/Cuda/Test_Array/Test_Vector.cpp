#include "gtest/gtest.h"
#include "Vector.h"

using namespace dyno;

TEST(Vector, cout)
{
	Vec2f vec2(1, 2);
	std::cout << vec2 << std::endl;

	Vec3f vec3(1, 2, 3);
	std::cout << vec3 << std::endl;

	Vec3f vec31(0, 3, 4);
	std::cout << vec31.norm()<<" "<<vec31.normSquared()<<" "<<vec31.dot(vec3) << std::endl;
	vec31=vec31.normalize();
	std::cout << vec31 << std::endl;

	Vec4f vec4(1, 2, 3, 4);
	std::cout << vec4 << std::endl;
}