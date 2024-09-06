#pragma once
#include "Module/ComputeModule.h"
#include "Primitive/Primitive3D.h"

#include "Catalyzer/VkReduce.h"
#include "Catalyzer/VkScan.h"

namespace dyno
{
// 	struct AlignedBox3D
// 	{
// 		dyno::Vec3f v0;
// 		dyno::Vec3f v1;
// 	};

	class CollisionDetectionBroadPhase : public ComputeModule
	{
	public:
		CollisionDetectionBroadPhase();
		virtual ~CollisionDetectionBroadPhase();

	public:
		float mGridSizeLimit;	// "Limit the smallest grid size";

		DEF_ARRAY_IN(AlignedBox3D, BoundingBox, DeviceType::GPU, "");
		//VkDeviceArray<AlignedBox3D> mAABB;

		//DEF_ARRAY_IN(AlignedBox3D, Other, DeviceType::GPU, "");
		//VkDeviceArray<AlignedBox3D> mOther;

		DEF_ARRAY_IN(uint32_t, CollisionMask, DeviceType::GPU, "");
		//VkDeviceArray<uint32_t>* mCollisionType;

		DEF_ARRAY_IN(uint32_t, ShapeType, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Vec2u, Contacts, DeviceType::GPU, "");
		//VkDeviceArray<Vec2u> mContactList;

	protected:
		void compute() override;

	private:
		std::shared_ptr<VkReduce<int>> vkr;

		std::shared_ptr<VkScan<int>> vks;

		DArray<int> counter;
		DArray<int> startIndex;
	};
}
