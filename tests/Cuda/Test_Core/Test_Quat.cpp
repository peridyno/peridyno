#include "gtest/gtest.h"
#include "Quat.h"
using namespace dyno;

TEST(Quat, func)
{
	Quat<float> q(0.0f, Vec3f(0.0f, 1.0f, 0.0f));

	Quat<float> quat;
	auto mat = quat.toMatrix3x3();

	auto mat4 = quat.toMatrix4x4();

	Quat<float> q2(0.2f, Vec3f(0.0f, 1.0f, 0.0f));

	float angle;
	Vec3f axis;
	q2.toRotationAxis(angle, axis);

	Quat<float> q3(0.21f, Vec3f(0.0f, 1.0f, 0.0f));

	Quat<float> q0(0.0f, 0.0f, 0.0f, 1.0f);
	Quat<float> omega(1.0f, 0.0f, 0.0f, 0.0f);

	auto qN = q0 + 0.5f*omega * q0;
	qN.normalize();

	EXPECT_EQ(abs(angle - 0.2f) < 10*EPSILON, true);
}

TEST(Quat, mat3)
{
	Quat<float> q(0.0f, 0.0f, 0.957910538f, 0.287067115f);
	auto mat3 = q.toMatrix3x3();
	Quat<float> q2(mat3);

	EXPECT_EQ((q2 - q).norm() < EPSILON, true);

	q = Quat<float>(0.5f, 0.6f, 0.9f, 1.0f);
	q.normalize();
	mat3 = q.toMatrix3x3();
	q2 = Quat<float>(mat3);

	EXPECT_EQ((q2 - q).norm() < EPSILON, true);
}