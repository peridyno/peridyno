#include "NeighborElementQuery.h"
#include "CollisionDetectionAlgorithm.h"

#include "Collision/CollisionDetectionBroadPhase.h"

#include "Topology/Primitive3D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborElementQuery, TDataType)
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

		boundary_expand = 0.0075f;

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
			//box.v0 -= boundary_expand;
			//box.v1 += boundary_expand;
			break;
		}
		case ET_TET:
		{
			
			box = tets[tId - elementOffset.tetIndex()].aabb();
			//box.v0 -= boundary_expand;
			//box.v1 += boundary_expand;

			/*if (tId % 100 == 0)
				printf("OK   %.3lf   %.3lf   %.3lf\n", box.v0[0], box.v0[1], box.v0[2]);*/

			break;
		}
		case ET_CAPSULE:
		{
			box = caps[tId - elementOffset.capsuleIndex()].aabb();
			//box.v0 -= boundary_expand;
			//box.v1 += boundary_expand;

			/*printf("inside!!!!!!!!  seg1 Pos\n%.3lf %.3lf %.3lf    %.3lf %.3lf %.3lf\n",
				caps[tId - elementOffset.capsuleIndex()].segment.v0[0],
				caps[tId - elementOffset.capsuleIndex()].segment.v0[1],
				caps[tId - elementOffset.capsuleIndex()].segment.v0[2],
				caps[tId - elementOffset.capsuleIndex()].segment.v1[0],
				caps[tId - elementOffset.capsuleIndex()].segment.v1[1],
				caps[tId - elementOffset.capsuleIndex()].segment.v1[2]
			);*/
			break;
		}
		case ET_TRI:
		{
			boundary_expand = 0.01;
			box = tris[tId - elementOffset.triangleIndex()].aabb();
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

		//printf("mask i = %d mask j = %d\n", mask_i, mask_j);

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
			//printf("OKKKK\n");
			//printf("OKKKK\n");
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
		
		else if (eleType_i == ET_TRI && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TRI, ET_SPHERE))
		{
			auto tri = triangles[ids.bodyId1 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId2];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, sphere, tri);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TRI))
		{
			auto tri = triangles[ids.bodyId2 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId1];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, sphere, tri);
			/*if (manifold.contactCount > 0 && manifold.contacts[0].penetration < -0.002f)
				printf("bb %.10lf\n", manifold.contacts[0].penetration);*/
		}
		else if (eleType_i == ET_TET && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_TET, ET_TRI))
		{
			auto tri = triangles[ids.bodyId2 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			
			CollisionDetection<Real>::request(manifold, tet, tri);
			//printf("aa %d\n", manifold.contactCount);
		}
		else if (eleType_i == ET_TRI && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TRI, ET_TET))
		{
			auto tri = triangles[ids.bodyId1 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			
			CollisionDetection<Real>::request(manifold, tri, tet);
			
		}
		else if (eleType_i == ET_TET && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_TET, ET_CAPSULE))
		{
			auto cap = caps[ids.bodyId2 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];

			CollisionDetection<Real>::request(manifold, tet, cap);
		}
		else if (eleType_j == ET_TET && eleType_i == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_TET))
		{
			auto cap = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];

			CollisionDetection<Real>::request(manifold, cap, tet);
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

		else if (eleType_i == ET_TRI && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TRI, ET_SPHERE))
		{
			//printf("aaaa\n");
			auto tri = tris[ids.bodyId1 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId2];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, tri, sphere);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TRI))
		{
			//printf("bbbb\n");
			auto tri = tris[ids.bodyId2 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId1];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, sphere, tri);
			
		}
		else if (eleType_i == ET_TET && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_TET, ET_TRI))
		{
			auto tri = tris[ids.bodyId2 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			//printf("inside!!~~~~\n");
			CollisionDetection<Real>::request(manifold, tet, tri);

			

			/*if (manifold.contactCount > 0)
				printf("inside!!!!!!!!  %d  %.10lf\n%.3lf   %.3lf    %.3lf\n",
					manifold.contactCount, manifold.contacts[0].penetration,
					manifold.normal[0],
					manifold.normal[1],
					manifold.normal[2]
				);*/
			//printf("inside!!\n");
		}
		else if (eleType_i == ET_TRI && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TRI, ET_TET))
		{
			auto tri = tris[ids.bodyId1 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			//printf("inside!!\n");
			CollisionDetection<Real>::request(manifold, tri, tet);

			//manifold.contacts[0].penetration -= 0.001f;

			/*if(manifold.contactCount > 0)
			printf("inside!!!!  %d  %.10lf\n%.3lf   %.3lf    %.3lf\n", 
				manifold.contactCount, manifold.contacts[0].penetration,
				manifold.normal[0],
				manifold.normal[1],
				manifold.normal[2]
				);*/
		}
		else if (eleType_i == ET_TET && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_TET, ET_CAPSULE))
		{
			auto cap = caps[ids.bodyId2 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];

			printf("!!!!!!aaaa\n");

			CollisionDetection<Real>::request(manifold, tet, cap);

			if (manifold.contactCount > 0)
				printf("inside!!!!!!!!  %d  %.10lf\n%.3lf   %.3lf    %.3lf\nseg1 Pos: %.3lf %.3lf %.3lf    %.3lf %.3lf %.3lf\n",
					manifold.contactCount, manifold.contacts[0].penetration,
					manifold.normal[0],
					manifold.normal[1],
					manifold.normal[2],
					cap.segment.v0[0],
					cap.segment.v0[1],
					cap.segment.v0[2],
					cap.segment.v1[0],
					cap.segment.v1[1],
					cap.segment.v1[2]
				);
		}
		else if (eleType_j == ET_TET && eleType_i == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_TET))
		{
			auto cap = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			//printf("!!!!!!bbbb\n");
			CollisionDetection<Real>::request(manifold, cap, tet);

			if (manifold.contactCount > 0)
				printf("inside bPart  %d  %.10lf\n%.3lf   %.3lf    %.3lf\nseg1 Pos: %.3lf %.3lf %.3lf    %.3lf %.3lf %.3lf\n",
					manifold.contactCount, manifold.contacts[0].penetration,
					manifold.normal[0],
					manifold.normal[1],
					manifold.normal[2],
					cap.segment.v0[0],
					cap.segment.v0[1],
					cap.segment.v0[2],
					cap.segment.v1[0],
					cap.segment.v1[1],
					cap.segment.v1[2]
				);
		}

		int offset = prefix[tId];
		for (int n = 0; n < manifold.contactCount; n++)
		{
			ContactPair cp;

			//if(abs(idx.body))

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

			//if(id.bodyId1 > index)

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
		
		int t_num = inTopo->totalSize();

		cnt++;
		//if(cnt % 5 != 1)
		if (cnt % 1 != 0)
		{
			printf("not nbq\n");
			return;
		}
		printf("nbq\n");

		if (t_num == 0)
		{
			auto& contacts = this->outContacts()->getData();
			
			contacts.resize(0);
			return;
		}
		
		if (m_queriedAABB.size() != t_num)
		{
			m_queriedAABB.resize(t_num);
		}
		if (m_queryAABB.size() != t_num)
		{
			m_queryAABB.resize(t_num);
		}
		//printf("=========== ============= INSIDE SELF COLLISION %d\n", t_num);
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


		m_broadPhaseCD->inSource()->assign(m_queryAABB);
		m_broadPhaseCD->inTarget()->assign(m_queriedAABB);
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
		contacts.reset();
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