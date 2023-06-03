#include "CollisionDetectionBroadPhase.h"
#include "Catalyzer/VkScan.h"
#include "Catalyzer/VkReduce.h"
#include "VkTransfer.h"

#define WORKGROUP_SIZE 64

namespace dyno
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
				CONSTANT(uint32_t))				//number of AABBs
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
				CONSTANT(uint32_t))				//number of AABBs
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
		uint num = this->inBoundingBox()->size();

		if (counter.size() != num){
			counter.resize(num);
			startIndex.resize(num);
		}

		VkConstant<uint32_t> constAABBSize(num);

		if (true)
		{
			kernel("CollisionCounterInBroadPhase")->begin();
			kernel("CollisionCounterInBroadPhase")->enqueue(
				vkDispatchSize(num, WORKGROUP_SIZE),
				counter.handle(),
				this->inBoundingBox()->getDataPtr()->handle(),
				this->inCollisionMask()->getDataPtr()->handle(),
				this->inShapeType()->getDataPtr()->handle(),
				&constAABBSize);
			kernel("CollisionCounterInBroadPhase")->end();
		}
		kernel("CollisionCounterInBroadPhase")->update();
		
		int totalSize = vkr->reduce(*counter.handle());

		if (totalSize <= 0) {
			this->outContacts()->clear();
			return;
		}

		vks->scan(*startIndex.handle(), *counter.handle(), EXCLUSIVESCAN);

		if (this->outContacts()->size() != totalSize) {
			this->outContacts()->resize(totalSize);
		}

		if (true)
		{
			kernel("CollisionDetectionInBroadPhase")->begin();
			kernel("CollisionDetectionInBroadPhase")->enqueue(
				vkDispatchSize(num, WORKGROUP_SIZE),
				this->outContacts()->getDataPtr()->handle(),
				startIndex.handle(),
				this->inBoundingBox()->getDataPtr()->handle(),
				this->inCollisionMask()->getDataPtr()->handle(),
				this->inShapeType()->getDataPtr()->handle(),
				&constAABBSize);
			kernel("CollisionDetectionInBroadPhase")->end();
		}
		kernel("CollisionDetectionInBroadPhase")->update();
	}

}