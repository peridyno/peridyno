#include "gtest/gtest.h"
#include "Collision/CollisionDetectionBroadPhase.h"

using namespace dyno;

TEST(BroadPhaseCollisionDetection, doCollision)
{
	std::vector<AABB> h_aabb_arr;
	h_aabb_arr.resize(3);

	h_aabb_arr[0].v0 = Vec3f(0.101f, 0.101f, 0.101f);
	h_aabb_arr[0].v1 = Vec3f(0.102f, 0.102f, 0.102f);

	h_aabb_arr[1].v0 = Vec3f(0.251f, 0.251f, 0.251f);
	h_aabb_arr[1].v1 = Vec3f(0.252f, 0.252f, 0.252f);

	h_aabb_arr[2].v0 = Vec3f(0.951f, 0.951f, 0.951f);
	h_aabb_arr[2].v1 = Vec3f(0.952f, 0.952f, 0.952f);


	std::vector<AABB> h_aabb_arr2;
	h_aabb_arr2.resize(1);

	h_aabb_arr2[0].v0 = Vec3f(0.1f, 0.1f, 0.1f);
	h_aabb_arr2[0].v1 = Vec3f(0.2f, 0.2f, 0.2f);

// 	h_aabb_arr2[1].v0 = Vec3f(0.25f, 0.25f, 0.25f);
// 	h_aabb_arr2[1].v1 = Vec3f(0.3f, 0.3f, 0.4f);

	CollisionDetectionBroadPhase<DataType3f> collisionModule;
	collisionModule.inSource()->setValue(h_aabb_arr);
	collisionModule.inTarget()->setValue(h_aabb_arr);


	collisionModule.update();
}