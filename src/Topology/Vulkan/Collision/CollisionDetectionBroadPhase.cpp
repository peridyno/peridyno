#include "CollisionDetectionBroadPhase.h"
#include "Catalyzer/VkScan.h"
#include "Catalyzer/VkReduce.h"
#include "VkTransfer.h"

#define WORKGROUP_SIZE 64

namespace px
{
	CollisionDetectionBroadPhase::CollisionDetectionBroadPhase()
		: dyno::CollisionModel()
	{
		vkr = std::make_shared<VkReduce<int>>();
		vks = std::make_shared<VkScan<int>>();

		this->addKernel(
			"CollisionCounterInBroadPhase",
			std::make_shared<VkProgram>(
				BUFFER(int),				//number collided AABBs
				BUFFER(AlignedBox3D),		//AABB
				BUFFER(uint32_t),			//collision type
				BUFFER(uint32_t),			//shape type
				UNIFORM(uint32_t))				//number of AABBs
		);
		kernel("CollisionCounterInBroadPhase")->load(getAssetPath() + "shaders/glsl/collision/CollisionCounterInBroadPhase.comp.spv");

		this->addKernel(
			"CollisionDetectionInBroadPhase",
			std::make_shared<VkProgram>(
				BUFFER(dyno::Vec2u),				//collision pairs
				BUFFER(int),				//start index
				BUFFER(AlignedBox3D),		//AABB
				BUFFER(uint32_t),			//collision type
				BUFFER(uint32_t),			//shape type
				UNIFORM(uint32_t))				//number of AABBs
		);
		kernel("CollisionDetectionInBroadPhase")->load(getAssetPath() + "shaders/glsl/collision/CollisionDetectionInBroadPhase.comp.spv");
	}

	CollisionDetectionBroadPhase::~CollisionDetectionBroadPhase()
	{
	}

	bool CollisionDetectionBroadPhase::initializeImpl()
	{
		return true;
	}

	void CollisionDetectionBroadPhase::doCollision()
	{
		bool rebuilding = VK_BUFFER_REALLOCATED == counter.resize(mAABB.size());
		startIndex.resize(mAABB.size());

		VkConstant<uint32_t> constAABBSize(mAABB.size());

		if (true)
		{
			kernel("CollisionCounterInBroadPhase")->begin();
			kernel("CollisionCounterInBroadPhase")->enqueue(
				vkDispatchSize(mAABB.size(), WORKGROUP_SIZE),
				&counter,
				&mAABB,
				mCollisionType,
				mShapeType,
				&constAABBSize);
			kernel("CollisionCounterInBroadPhase")->end();
		}
		kernel("CollisionCounterInBroadPhase")->update();
		
		int totalSize = vkr->reduce(counter);

		if (totalSize <= 0) {
			mContactList.reset();
			return;
		}

		vks->scan(startIndex, counter, EXCLUSIVESCAN);

		rebuilding |= VK_BUFFER_REALLOCATED == mContactList.resize(totalSize);

		if (true)
		{
			kernel("CollisionDetectionInBroadPhase")->begin();
			kernel("CollisionDetectionInBroadPhase")->enqueue(
				vkDispatchSize(mAABB.size(), WORKGROUP_SIZE),
				&mContactList,
				&startIndex,
				&mAABB,
				mCollisionType,
				mShapeType,
				&constAABBSize);
			kernel("CollisionDetectionInBroadPhase")->end();
		}
		kernel("CollisionDetectionInBroadPhase")->update();
	}

}