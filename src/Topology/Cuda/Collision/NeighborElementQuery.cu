#include "NeighborElementQuery.h"
#include "CollisionDetectionAlgorithm.h"

#include "Collision/CollisionDetectionBroadPhase.h"

#include "Primitive/Primitive3D.h"

#include "Timer.h"

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborElementQuery, TDataType)
		typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	struct ContactId
	{
		int bodyId1 = INVLIDA_ID;
		int bodyId2 = INVLIDA_ID;
	};

	template<typename TDataType>
	NeighborElementQuery<TDataType>::NeighborElementQuery()
		: ComputeModule()
	{
		this->inCollisionMask()->tagOptional(true);

		this->inAttribute()->tagOptional(true);

		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();

		this->varGridSizeLimit()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					mBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
				})
		);

		this->varSelfCollision()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					mBroadPhaseCD->varSelfCollision()->setValue(this->varSelfCollision()->getValue());
				})
		);

		this->varGridSizeLimit()->setValue(Real(0.011));
		this->varSelfCollision()->setValue(true);
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
			break;
		}
		case ET_TET:
		{
			
			box = tets[tId - elementOffset.tetIndex()].aabb();
			break;
		}
		case ET_CAPSULE:
		{
			box = caps[tId - elementOffset.capsuleIndex()].aabb();
			break;
		}
		case ET_TRI:
		{
			boundary_expand = 0.01;
			box = tris[tId - elementOffset.triangleIndex()].aabb();
			break;
		}
		default:
			break;
		}

		box.v0 -= boundary_expand;
		box.v1 += boundary_expand;

		boundingBox[tId] = box;
	}

	__device__ inline bool checkCollision(CollisionMask cType0, CollisionMask cType1, ElementType eleType0, ElementType eleType1)
	{
		bool canCollide = (cType0 & eleType1) != 0 && (cType1 & eleType0) > 0;
		if (!canCollide)
			return false;

		return true;
	}

	template<typename Box3D, typename ContactPair>
	__global__ void NEQ_Narrow_Count(
		DArray<int> count,
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
		DArray<Triangle3D> triangles,
		DArray<Attribute> attribute,
		ElementOffset elementOffset,
		Real dHat,
		bool enableSelfCollision,
		bool enableCollisionMask)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;

		ContactId ids = nbr[tId];
		ElementType eleType_i = elementOffset.checkElementType(ids.bodyId1);
		ElementType eleType_j = elementOffset.checkElementType(ids.bodyId2);

		CollisionMask mask_i = enableCollisionMask ? mask[ids.bodyId1] : CollisionMask::CT_AllObjects;
		CollisionMask mask_j = enableCollisionMask ? mask[ids.bodyId2] : CollisionMask::CT_AllObjects;

		TManifold<Real> manifold;

		if (!enableSelfCollision)
		{
			Attribute att_i = attribute[ids.bodyId1];
			Attribute att_j = attribute[ids.bodyId2];

			printf("Collision excluded \n");

			if (att_i.objectId() == att_j.objectId())
				return;
		}

		if (eleType_i == ET_BOX && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_BOX, ET_BOX))
		{
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];

			boxA.extent += dHat;
			boxB.extent += dHat;

			CollisionDetection<Real>::request(manifold, boxA, boxB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_SPHERE, ET_BOX))
		{
			auto sphere = spheres[ids.bodyId1];
			auto box = boxes[ids.bodyId2 - elementOffset.boxIndex()];

			sphere.radius += dHat;
			box.extent += dHat;

			CollisionDetection<Real>::request(manifold, sphere, box);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_BOX, ET_SPHERE))
		{
			auto box = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto sphere = spheres[ids.bodyId2];

			box.extent += dHat;
			sphere.radius += dHat;

			CollisionDetection<Real>::request(manifold, box, sphere);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_SPHERE, ET_SPHERE))
		{
			auto sphereA = spheres[ids.bodyId1];
			auto sphereB = spheres[ids.bodyId2];

			sphereA.radius += dHat;
			sphereB.radius += dHat;

			CollisionDetection<Real>::request(manifold, sphereA, sphereB);
		}
		else if(eleType_i == ET_TET && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TET, ET_TET))
		{
			//TODO: consider dHat
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, tetA, tetB);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_BOX && checkCollision(mask_i, mask_j, ET_TET, ET_BOX))
		{
			//TODO: consider dHat
			auto tetA = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto boxB = boxes[ids.bodyId2 - elementOffset.boxIndex()];
			CollisionDetection<Real>::request(manifold, tetA, boxB);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_BOX, ET_TET))
		{
			//TODO: consider dHat
			auto boxA = boxes[ids.bodyId1 - elementOffset.boxIndex()];
			auto tetB = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, boxA, tetB);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TET))
		{
			//TODO: consider dHat
			auto sphere = spheres[ids.bodyId1];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			CollisionDetection<Real>::request(manifold, sphere, tet);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TET, ET_SPHERE))
		{
			//TODO: consider dHat
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			auto sphere = spheres[ids.bodyId2];
			CollisionDetection<Real>::request(manifold, tet, sphere);
		}
		else if (eleType_i == ET_TRI && eleType_j == ET_SPHERE && checkCollision(mask_i, mask_j, ET_TRI, ET_SPHERE))
		{
			//TODO: consider dHat
			auto tri = triangles[ids.bodyId1 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId2];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, sphere, tri);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_SPHERE, ET_TRI))
		{
			//TODO: consider dHat
			auto tri = triangles[ids.bodyId2 - elementOffset.triangleIndex()];
			auto sphere = spheres[ids.bodyId1];
			sphere.radius += 0.01f;
			CollisionDetection<Real>::request(manifold, sphere, tri);
		}
		else if (eleType_i == ET_TET && eleType_j == ET_TRI && checkCollision(mask_i, mask_j, ET_TET, ET_TRI))
		{
			//TODO: consider dHat
			auto tri = triangles[ids.bodyId2 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];
			
			CollisionDetection<Real>::request(manifold, tet, tri);
		}
		else if (eleType_i == ET_TRI && eleType_j == ET_TET && checkCollision(mask_i, mask_j, ET_TRI, ET_TET))
		{
		//TODO: consider dHat
			auto tri = triangles[ids.bodyId1 - elementOffset.triangleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];
			
			CollisionDetection<Real>::request(manifold, tri, tet);
		}
		//Capsule with other primitives
		else if (eleType_i == ET_TET && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_TET, ET_CAPSULE))		//tet-capsule
		{
			//TODO: consider dHat
			auto cap = caps[ids.bodyId2 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId1 - elementOffset.tetIndex()];

			CollisionDetection<Real>::request(manifold, tet, cap);
		}
		else if (eleType_j == ET_TET && eleType_i == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_TET))
		{
			//TODO: consider dHat
			auto cap = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto tet = tets[ids.bodyId2 - elementOffset.tetIndex()];

			CollisionDetection<Real>::request(manifold, cap, tet);
		}
		else if (eleType_i == ET_CAPSULE && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_CAPSULE))	//capsule-capsule
		{
			auto cap1 = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto cap2 = caps[ids.bodyId2 - elementOffset.capsuleIndex()];

			cap1.radius += dHat;
			cap2.radius += dHat;

			CollisionDetection<Real>::request(manifold, cap1, cap2);
		}
		else if (eleType_i == ET_BOX && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_BOX, ET_CAPSULE))			//box-capsule
		{
			auto cap = caps[ids.bodyId2 - elementOffset.capsuleIndex()];
			auto box = boxes[ids.bodyId1 - elementOffset.boxIndex()];

			cap.radius += dHat;
			box.extent += dHat;

			CollisionDetection<Real>::request(manifold, box, cap);
		}
		else if (eleType_j == ET_BOX && eleType_i == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_BOX))
		{
			auto cap = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto box = boxes[ids.bodyId2 - elementOffset.boxIndex()];

			cap.radius += dHat;
			box.extent += dHat;

			CollisionDetection<Real>::request(manifold, cap, box);
		}
		else if (eleType_i == ET_SPHERE && eleType_j == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_SPHERE, ET_CAPSULE))			//sphere-capsule
		{
			auto cap = caps[ids.bodyId2 - elementOffset.capsuleIndex()];
			auto sphere = spheres[ids.bodyId1 - elementOffset.sphereIndex()];

			cap.radius += dHat;
			sphere.radius += dHat;

			CollisionDetection<Real>::request(manifold, sphere, cap);
		}
		else if (eleType_j == ET_SPHERE && eleType_i == ET_CAPSULE && checkCollision(mask_i, mask_j, ET_CAPSULE, ET_SPHERE))
		{
			auto cap = caps[ids.bodyId1 - elementOffset.capsuleIndex()];
			auto sphere = spheres[ids.bodyId2 - elementOffset.sphereIndex()];

			cap.radius += dHat;
			sphere.radius += dHat;

			CollisionDetection<Real>::request(manifold, cap, sphere);
		}
		
		count[tId] = manifold.contactCount;

		int offset = 8 * tId;
		for (int n = 0; n < manifold.contactCount; n++)
		{
			ContactPair cp;
			cp.pos1 = manifold.contacts[n].position + dHat * manifold.normal;
			cp.pos2 = manifold.contacts[n].position + dHat * manifold.normal;
			cp.normal1 = -manifold.normal;
			cp.normal2 = manifold.normal;
			cp.bodyId1 = ids.bodyId1;
			cp.bodyId2 = ids.bodyId2;
			cp.contactType = ContactType::CT_NONPENETRATION;
			cp.interpenetration = -manifold.contacts[n].penetration - 2 * dHat;
			nbr_cons[offset + n] = cp;
		}
	}

	template<typename ContactPair>
	__global__ void NEQ_Narrow_Set(
		DArray<ContactPair> nbr_cons,
		DArray<ContactPair> nbr_cons_all,
		DArray<ContactId> nbr,
		DArray<int> prefix,
		DArray<int> sum)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;

		ContactId ids = nbr[tId];

		int offset = prefix[tId];
		int size = sum[tId];
		for (int n = 0; n < size; n++)
		{
			nbr_cons[offset + n] = nbr_cons_all[8 * tId + n];
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

		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();

		int t_num = inTopo->totalSize();

		if (t_num == 0)
		{
			auto& contacts = this->outContacts()->getData();
			contacts.resize(0);
			return;
		}

		if (mQueriedAABB.size() != t_num){
			mQueriedAABB.resize(t_num);
		}

		if (mQueryAABB.size() != t_num){
			mQueryAABB.resize(t_num);
		}
		//printf("=========== ============= INSIDE SELF COLLISION %d\n", t_num);
		ElementOffset elementOffset = inTopo->calculateElementOffset();

		Real dHat = this->varDHead()->getValue();

		cuExecute(t_num,
			NEQ_SetupAABB,
			mQueriedAABB,
			inTopo->getBoxes(),
			inTopo->getSpheres(),
			inTopo->getTets(),
			inTopo->getCaps(),
			inTopo->getTris(),
			elementOffset,
			dHat);

		mQueryAABB.assign(mQueriedAABB);

		mBroadPhaseCD->inSource()->assign(mQueryAABB);
		mBroadPhaseCD->inTarget()->assign(mQueriedAABB);
		// 
		mBroadPhaseCD->update();

		auto& contactList = mBroadPhaseCD->outContactList()->getData();
		if (contactList.size() == 0) return;

		DArray<int> count(contactList.size());
		cuExecute(contactList.size(),
			CCL_CountListSize,
			count,
			contactList);

		int totalSize = mReduce.accumulate(count.begin(), count.size());

		if (totalSize <= 0) {
			this->outContacts()->clear();
			return;
		}

		mScan.exclusive(count);

		DArray<ContactId> deviceIds(totalSize);

		cuExecute(contactList.size(),
			CCL_SetupContactIds,
			deviceIds,
			count,
			contactList);

		count.clear();

		Real zero = 0;

		DArray<int> contactNum;
		DArray<int> contactNumCpy;

		contactNum.resize(deviceIds.size());
		contactNum.reset();
		contactNumCpy.resize(deviceIds.size());
		contactNumCpy.reset();

		DArray<TContactPair<Real>> nbr_cons_tmp;
		nbr_cons_tmp.resize(deviceIds.size() * 8);
		nbr_cons_tmp.reset();

		if (this->inCollisionMask()->isEmpty())
		{
			DArray<CollisionMask> dummyCollisionMask;
			if (!this->varSelfCollision()->getValue() && !this->inAttribute()->isEmpty())
			{
				cuExecute(deviceIds.size(),
					NEQ_Narrow_Count,
					contactNum,
					nbr_cons_tmp,
					deviceIds,
					dummyCollisionMask,
					inTopo->getBoxes(),
					inTopo->getSpheres(),
					inTopo->getTets(),
					inTopo->getTetSDF(),
					inTopo->getTetBodyMapping(),
					inTopo->getTetElementMapping(),
					inTopo->getCaps(),
					inTopo->getTris(),
					this->inAttribute()->getData(),
					elementOffset,
					dHat,
					false,
					false);
			}
			else
			{
				DArray<Attribute> dummyAttribute;
				cuExecute(deviceIds.size(),
					NEQ_Narrow_Count,
					contactNum,
					nbr_cons_tmp,
					deviceIds,
					dummyCollisionMask,
					inTopo->getBoxes(),
					inTopo->getSpheres(),
					inTopo->getTets(),
					inTopo->getTetSDF(),
					inTopo->getTetBodyMapping(),
					inTopo->getTetElementMapping(),
					inTopo->getCaps(),
					inTopo->getTris(),
					dummyAttribute,
					elementOffset,
					dHat,
					true,
					false);
			}

		}
		else
		{
			if (!this->varSelfCollision()->getValue() && !this->inAttribute()->isEmpty())
			{
				cuExecute(deviceIds.size(),
					NEQ_Narrow_Count,
					contactNum,
					nbr_cons_tmp,
					deviceIds,
					this->inCollisionMask()->getData(),
					inTopo->getBoxes(),
					inTopo->getSpheres(),
					inTopo->getTets(),
					inTopo->getTetSDF(),
					inTopo->getTetBodyMapping(),
					inTopo->getTetElementMapping(),
					inTopo->getCaps(),
					inTopo->getTris(),
					this->inAttribute()->getData(),
					elementOffset,
					dHat,
					false,
					true);
			}
			else
			{
				DArray<Attribute> dummyAttribute;
				cuExecute(deviceIds.size(),
					NEQ_Narrow_Count,
					contactNum,
					nbr_cons_tmp,
					deviceIds,
					this->inCollisionMask()->getData(),
					inTopo->getBoxes(),
					inTopo->getSpheres(),
					inTopo->getTets(),
					inTopo->getTetSDF(),
					inTopo->getTetBodyMapping(),
					inTopo->getTetElementMapping(),
					inTopo->getCaps(),
					inTopo->getTris(),
					dummyAttribute,
					elementOffset,
					dHat,
					true,
					true);
			}
		}

		contactNumCpy.assign(contactNum);
		
		int sum = mReduce.accumulate(contactNum.begin(), contactNum.size());

		auto& contacts = this->outContacts()->getData();
		mScan.exclusive(contactNum, true);
		contacts.resize(sum);
		contacts.reset();
		if (sum > 0)
		{
			cuExecute(deviceIds.size(),
				NEQ_Narrow_Set,
				contacts,
				nbr_cons_tmp,
				deviceIds,
				contactNum,
				contactNumCpy);
		}
		contactNumCpy.clear();
		contactNum.clear();
		deviceIds.clear();
		nbr_cons_tmp.clear();
	}

	DEFINE_CLASS(NeighborElementQuery);
}