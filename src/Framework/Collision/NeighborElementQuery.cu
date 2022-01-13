#include "NeighborElementQuery.h"
#include "CollisionDetectionAlgorithm.h"

#include "Collision/CollisionDetectionBroadPhase.h"

#include "Topology/Primitive3D.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(NeighborElementQuery, TDataType)
		typedef typename TOrientedBox3D<Real> Box3D;

	struct ContactId
	{
		int bodyId1 = INVLIDA_ID;
		int bodyId2 = INVLIDA_ID;
	};

	template<typename TDataType>
	NeighborElementQuery<TDataType>::NeighborElementQuery()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));

		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
		//fout.open("data_Oct_without_arrange.txt");
	}

	template<typename TDataType>
	NeighborElementQuery<TDataType>::~NeighborElementQuery()
	{
	}

	template<typename Real, typename Coord>
	__global__ void NEQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> position,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		AABB box;
		Coord p = position[pId];
		box.v0 = p - radius;
		box.v1 = p + radius;

		boundingBox[pId] = box;
	}

	template<typename Box3D>
	__global__ void NEQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset,
		Real boundary_expand)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		ElementType eleType = elementOffset.checkElementType(tId);

		//Real boundary_expand = 0.0075f;

		AABB box;
		switch (eleType)
		{
		case ET_SPHERE:
		{
			box = spheres[tId].aabb();

			break;
		}
		case ET_BOX:
		{
			box = boxes[tId - elementOffset.boxIndex()].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case ET_TET:
		{
			box = tets[tId - elementOffset.tetIndex()].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case ET_CAPSULE:
		{
			box = caps[tId - elementOffset.capsuleIndex()].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case ET_TRI:
		{
			box = tris[tId - elementOffset.triangleIndex()].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			/*printf("%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\n=========\n",
				box.v0[0], box.v0[1], box.v0[2],
				box.v1[0], box.v1[1], box.v1[2]);*/
			break;
		}
		default:
			break;
		}

		boundingBox[tId] = box;
	}

	__device__ inline bool checkCollision(CollisionMask cType0, CollisionMask cType1, ElementType eleType0, ElementType eleType1)
	{
		bool canCollide = (cType0 & eleType1) != 0 && (cType1 & eleType0) > 0;
		if (!canCollide)
			return false;

		return true;
	}

	template<typename Box3D>
	__global__ void NEQ_Narrow_Count(
		DArray<int> count,
		DArray<ContactId> nbr,
		DArray<CollisionMask> mask,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Real> tets_sdf,
		DArray<int> tet_body_ids,
		DArray<TopologyModule::Tetrahedron> tet_element_ids,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> triangles,
		ElementOffset elementOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;

		ContactId ids = nbr[tId];
		ElementType eleType_i = elementOffset.checkElementType(ids.bodyId1);
		ElementType eleType_j = elementOffset.checkElementType(ids.bodyId2);

		CollisionMask mask_i = mask[ids.bodyId1];
		CollisionMask mask_j = mask[ids.bodyId2];

		TManifold<Real> manifold;
		if (eleType_i == ET_BOX && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_BOX, ET_BOX))
		{
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, boxA, boxB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_SPHERE, ET_BOX))
		{
			auto sphere = spheres[ids.bodyId1];
			auto box = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, sphere, box);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_BOX, ET_SPHERE))
		{
			auto box = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto sphere = spheres[ids.bodyId2];
			CollisionDetection<Real>::request(manifold, box, sphere);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_SPHERE, ET_SPHERE))
		{
			CollisionDetection<Real>::request(manifold, spheres[ids.bodyId1], spheres[ids.bodyId2]);
		}
		else if(eleType_i == ET_TET && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TET, ET_TET))
		{
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, tetA, tetB);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_TET, ET_BOX))
		{
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, tetA, boxB);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_BOX, ET_TET))
		{
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, boxA, tetB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TET))
		{
			auto sphere = spheres[ids.bodyId1];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, sphere, tet);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TET, ET_SPHERE))
		{
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto sphere = spheres[ids.bodyId2];
			CollisionDetection<Real>::request(manifold, tet, sphere);
		}

		count[tId] = manifold.contactCount;
	}

	template<typename Box3D, typename ContactPair>
	__global__ void NEQ_Narrow_Set(
		DArray<ContactPair> nbr_cons,
		DArray<ContactId> nbr,
		DArray<CollisionMask> mask,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Real> tets_sdf,
		DArray<int> tet_body_ids,
		DArray<TopologyModule::Tetrahedron> tet_element_ids,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		DArray<int> prefix,
		ElementOffset elementOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;
		
		ContactId ids = nbr[tId];
		ElementType eleType_i = elementOffset.checkElementType(ids.bodyId1);
		ElementType eleType_j = elementOffset.checkElementType(ids.bodyId2);

		CollisionMask mask_i = mask[ids.bodyId1];
		CollisionMask mask_j = mask[ids.bodyId2];

		TManifold<Real> manifold;
		if (eleType_i == ET_BOX && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_BOX, ET_BOX))
		{
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, boxA, boxB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_SPHERE, ET_BOX))
		{
			auto sphere = spheres[ids.bodyId1];
			auto box = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, sphere, box);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_BOX, ET_SPHERE))
		{
			auto box = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto sphere = spheres[ids.bodyId2];
			CollisionDetection<Real>::request(manifold, box, sphere);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_SPHERE, ET_SPHERE))
		{
			CollisionDetection<Real>::request(manifold, spheres[ids.bodyId1], spheres[ids.bodyId2]);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TET, ET_TET))
		{
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, tetA, tetB);
			
			/*printf("%.3lf %.3lf %.3lf  %.6lf\n", manifold.normal[0], manifold.normal[1], manifold.normal[2], 
				manifold.contacts[0].penetration);*/
		}
		else if (eleType_i == ET_TET && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_TET, ET_BOX))
		{
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, tetA, boxB);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_BOX, ET_TET))
		{
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, boxA, tetB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TET))
		{
			auto sphere = spheres[ids.bodyId1];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, sphere, tet);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TET, ET_SPHERE))
		{
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto sphere = spheres[ids.bodyId2];
			CollisionDetection<Real>::request(manifold, tet, sphere);
		}

		int offset = prefix[tId];
		for (int n = 0; n < manifold.contactCount; n++)
		{
			ContactPair cp;

			cp.pos1 = manifold.contacts[n].position;
			cp.pos2 = manifold.contacts[n].position;
			cp.normal1 = -manifold.normal;
			cp.normal2 = manifold.normal;
			cp.bodyId1 = ids.bodyId1;
			cp.bodyId2 = ids.bodyId2;
			cp.contactType = ContactType::CT_NONPENETRATION;
			cp.interpenetration = -manifold.contacts[n].penetration;
			nbr_cons[offset + n] = cp;
		}
	}

	__global__ void CCL_CountListSize(
		DArray<int> num,
		DArrayList<int> contactList)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		num[tId] = contactList[tId].size();
	}

	__global__ void CCL_SetupContactIds(
		DArray<ContactId> ids,
		DArray<int> index,
		DArrayList<int> contactList)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		int base = index[tId];

		auto& list_i = contactList[tId];
		for (int j = 0; j < list_i.size(); j++)
		{
			ContactId id;
			id.bodyId1 = tId;
			id.bodyId2 = list_i[j];

			ids[base + j] = id;
		}
	}

	template<typename TDataType>
	void NeighborElementQuery<TDataType>::compute()
	{
		auto inTopo = this->inDiscreteElements()->getDataPtr();
		auto& inMask = this->inCollisionMask()->getData();

		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();

		Real boundary_expand = 0.0f;
		//printf("=========== ============= INSIDE SELF COLLISION %d\n", discreteSet->getTets().size());
		int t_num = inTopo->totalSize();
		if (m_queriedAABB.size() != t_num)
		{
			m_queriedAABB.resize(t_num);
		}
		if (m_queryAABB.size() != t_num)
		{
			m_queryAABB.resize(t_num);
		}

		ElementOffset elementOffset = inTopo->calculateElementOffset();

		cuExecute(t_num,
			NEQ_SetupAABB,
			m_queriedAABB,
			inTopo->getBoxes(),
			inTopo->getSpheres(),
			inTopo->getTets(),
			inTopo->getCaps(),
			inTopo->getTris(),
			elementOffset,
			boundary_expand);

		m_queryAABB.assign(m_queriedAABB);


		Real radius = this->inRadius()->getData();

		m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
		m_broadPhaseCD->setSelfCollision(true);


		m_broadPhaseCD->inSource()->setValue(m_queryAABB);
		m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
		// 
		m_broadPhaseCD->update();


		auto& contactList = m_broadPhaseCD->outContactList()->getData();

		DArray<int> count(contactList.size());
		cuExecute(contactList.size(),
			CCL_CountListSize,
			count,
			contactList);

		int totalSize = m_reduce.accumulate(count.begin(), count.size());

		if (totalSize <= 0)
			return;

		m_scan.exclusive(count);

		DArray<ContactId> deviceIds(totalSize);

		cuExecute(contactList.size(),
			CCL_SetupContactIds,
			deviceIds,
			count,
			contactList);

		count.clear();

		Real zero = 0;

		DArray<int> contactNum;

		contactNum.resize(deviceIds.size());
		contactNum.reset();

		cuExecute(deviceIds.size(),
			NEQ_Narrow_Count,
			contactNum,
			deviceIds,
			inMask,
			inTopo->getBoxes(),
			inTopo->getSpheres(),
			inTopo->getTets(),
			inTopo->getTetSDF(),
			inTopo->getTetBodyMapping(),
			inTopo->getTetElementMapping(),
			inTopo->getCaps(),
			inTopo->getTris(),
			elementOffset);

		int sum = m_reduce.accumulate(contactNum.begin(), contactNum.size());

		auto& contacts = this->outContacts()->getData();
		m_scan.exclusive(contactNum, true);
		contacts.resize(sum);
		if (sum > 0)
		{
			cuExecute(deviceIds.size(),
				NEQ_Narrow_Set,
				contacts,
				deviceIds,
				inMask,
				inTopo->getBoxes(),
				inTopo->getSpheres(),
				inTopo->getTets(),
				inTopo->getTetSDF(),
				inTopo->getTetBodyMapping(),
				inTopo->getTetElementMapping(),
				inTopo->getCaps(),
				inTopo->getTris(),
				contactNum,
				elementOffset);
		}

		contactNum.clear();
		deviceIds.clear();
	}

	DEFINE_CLASS(NeighborElementQuery);
}