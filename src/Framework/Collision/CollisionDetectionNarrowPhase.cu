#include "CollisionDetectionNarrowPhase.h"
#include "Node.h"
#include "Topology/SparseOctree.h"
#include "Topology/DiscreteElements.h"

#include "CollisionDetectionBroadPhase.h"


namespace dyno
{
	IMPLEMENT_CLASS_1(CollisionDetectionNarrowPhase, TDataType)

	template<typename TDataType>
	CollisionDetectionNarrowPhase<TDataType>::CollisionDetectionNarrowPhase()
		: CollisionModel()
	{
		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	CollisionDetectionNarrowPhase<TDataType>::~CollisionDetectionNarrowPhase()
	{
	}


	template<typename TShape>
	__global__ void CD_ConstructAABB(DArray<AABB> abox,
		DArray<TShape> shapes,
		int shift)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= shapes.size()) return;

		abox[shift + tId] = shapes[tId].aabb();
	}

	template<typename TDataType>
	void CollisionDetectionNarrowPhase<TDataType>::doCollision()
	{
		Node* parent = this->getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return;
		}

		auto elements = TypeInfo::cast<DiscreteElements<TDataType>>(parent->getTopologyModule());
		if (elements == nullptr)
		{
			Log::sendMessage(Log::Error, "CollisionDetection: The topology module is not supported!");
			return;
		}

		auto boxes = elements->getBoxes();
		auto spheres = elements->getSpheres();

		int total_num = 0;
		total_num += boxes.size() + spheres.size();


		auto& aabb = m_broadPhaseCD->inSource()->getData();

		if (aabb.size() != total_num)
		{
			aabb.resize(total_num);
		}

		int shift = 0;

		cuExecute(boxes.size(),
			CD_ConstructAABB,
			aabb,
			boxes,
			shift);
		shift += boxes.size();

		cuExecute(spheres.size(),
			CD_ConstructAABB,
			aabb,
			spheres,
			shift);
		
		m_broadPhaseCD->update();

	}


	template<typename TDataType>
	bool CollisionDetectionNarrowPhase<TDataType>::isSupport(std::shared_ptr<CollidableObject> obj)
	{
		return true;
	}

	DEFINE_CLASS(CollisionDetectionNarrowPhase);
}