#pragma once
#include "Module/CollisionModel.h"

#include "Catalyzer/VkReduce.h"
#include "Catalyzer/VkScan.h"

namespace px
{
	struct AlignedBox3D
	{
		dyno::Vec3f v0;
		dyno::Vec3f v1;
	};

	class CollisionDetectionBroadPhase : public dyno::CollisionModel
	{
	public:
		CollisionDetectionBroadPhase();
		virtual ~CollisionDetectionBroadPhase();

		bool initializeImpl() override;

		void doCollision() override;

	public:
		float mGridSizeLimit;	// "Limit the smallest grid size";

		VkDeviceArray<AlignedBox3D> mAABB;

		VkDeviceArray<AlignedBox3D> mOther;

		VkDeviceArray<uint32_t>* mCollisionType;
		VkDeviceArray<uint32_t>* mShapeType;

		VkDeviceArray<dyno::Vec2u> mContactList;

	private:
		std::shared_ptr<VkReduce<int>> vkr;

		std::shared_ptr<VkScan<int>> vks;

		VkDeviceArray<int> counter;
		VkDeviceArray<int> startIndex;
	};
}
