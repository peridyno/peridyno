#include "CollistionDetectionTriangleSet.h"

#include "Primitive/Primitive3D.h"
#include "Topology/DiscreteElements.h"

namespace dyno
{
	template<typename TDataType>
	CollistionDetectionTriangleSet<TDataType>::CollistionDetectionTriangleSet()
		: ComputeModule()
	{
		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	CollistionDetectionTriangleSet<TDataType>::~CollistionDetectionTriangleSet()
	{

	}

	template<typename Coord, typename AABB>
	__global__ void CDTS_SetupAABBs(
		DArray<AABB> aabbs,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indices.size()) return;

		TopologyModule::Triangle index = indices[tId];

		Coord v0 = vertices[index[0]];
		Coord v1 = vertices[index[1]];
		Coord v2 = vertices[index[2]];

		aabbs[tId] = TTriangle3D<Real>(v0, v1, v2).aabb();
	}

	template<typename AABB, typename Box3D>
	__global__ void CDTS_SetupAABBForRigidBodies(
		DArray<AABB> boundingBox,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		ElementOffset elementOffset)
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
		default:
			break;
		}

		boundingBox[tId] = box;
	}

	template<typename Coord, typename Box3D, typename ContactPair>
	__global__ void CDTS_CountContacts(
		DArray<int> count,
		DArray<ContactPair> nbr_cons,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> capsules,
		DArray<Coord> vertices,		//triangle vertices
		DArray<TopologyModule::Triangle> indices,	//triangle indices
		DArrayList<int> contactListBroadPhase,
		DArray<Pair<uint, uint>> mapping,
		ElementOffset elementOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		auto& list_i = contactListBroadPhase[tId];

		ElementType eleType = elementOffset.checkElementType(tId);

		uint num = 0;
		Real d_hat = 0;
		Coord points[8];

		if (eleType == ET_BOX)
		{
			num = 8;
			Box3D box = boxes[tId - elementOffset.boxIndex()];
			Coord center = box.center;
			Coord u = box.u;
			Coord v = box.v;
			Coord w = box.w;
			Coord extent = box.extent;

			d_hat = 0;
			points[0] = center - u * extent[0] - v * extent[1] - w * extent[2];
			points[1] = center - u * extent[0] - v * extent[1] + w * extent[2];
			points[2] = center - u * extent[0] + v * extent[1] - w * extent[2];
			points[3] = center - u * extent[0] + v * extent[1] + w * extent[2];
			points[4] = center + u * extent[0] - v * extent[1] - w * extent[2];
			points[5] = center + u * extent[0] - v * extent[1] + w * extent[2];
			points[6] = center + u * extent[0] + v * extent[1] - w * extent[2];
			points[7] = center + u * extent[0] + v * extent[1] + w * extent[2];
		}
		else if (eleType == ET_SPHERE)
		{
			num = 1;
			Sphere3D sp = spheres[tId - elementOffset.sphereIndex()];

			d_hat = sp.radius;
			points[0] = sp.center;
		}
		else if (eleType == ET_CAPSULE)
		{
			num = 2;

			Capsule3D cap = capsules[tId - elementOffset.capsuleIndex()];

			d_hat = cap.radius;
			points[0] = cap.startPoint();
			points[1] = cap.endPoint();
		}
		else if (eleType == ET_TET)
		{
			num = 4;
			Tet3D tet = tets[tId - elementOffset.tetIndex()];
			
			d_hat = 0;
			points[0] = tet.v[0];
			points[1] = tet.v[1];
			points[2] = tet.v[2];
			points[3] = tet.v[3];
		}

		auto PROJECT_INSIDE = [](const TPoint3D<Real> p, const TTriangle3D<Real> triangle) -> bool
		{
			Real epsilon = Real(0.00001);
			TPlane3D<Real> plane(triangle.v[0], triangle.normal());
			
			TPoint3D<Real> proj = p.project(plane);

			typename TTriangle3D<Real>::Param tParam;
			bool bValid = triangle.computeBarycentrics(proj.origin, tParam);
			if (bValid)
			{
				return tParam.u > Real(0) - epsilon && tParam.u < Real(1) + epsilon && tParam.v > Real(0) - epsilon && tParam.v < Real(1) + epsilon && tParam.w > Real(0) - epsilon && tParam.w < Real(1) + epsilon;
			}
			else
			{
				return false;
			}
		};
		
		uint cnt = 0;
		for (uint i = 0; i < num; i++)
		{
			bool intersected = false;
			Real d_min = -REAL_MAX;
			Coord proj_min;
			Coord normal_min;

			TPoint3D<Real> pi = TPoint3D<Real>(points[i]);
			for (uint j = 0; j < list_i.size(); j++)
			{
				TopologyModule::Triangle index = indices[list_i[j]];
				TTriangle3D<Real> tj(vertices[index[0]], vertices[index[1]], vertices[index[2]]);

				Coord nj = tj.normal();

				if (PROJECT_INSIDE(pi, tj))
				{
					TPoint3D<Real> proj = pi.project(tj);
					Real d = nj.dot(pi.origin - proj.origin) - d_hat;

					if (d < 0)
					{
						//Find the closest triangle
						if (d > d_min)
						{
							d_min = d;
							proj_min = proj.origin;
							normal_min = nj;

							intersected = true;
						}
					}
				}
			}

			if (intersected)
			{
				ContactPair contact;
				contact.bodyId1 = mapping[tId].second;
				contact.bodyId2 = -1;
				contact.normal1 = normal_min;
				contact.pos1 = proj_min + normal_min * d_min;
				contact.contactType = ContactType::CT_BOUDNARY;
				contact.interpenetration = -d_min;

				nbr_cons[8 * tId + cnt] = contact;

				cnt++;
			}
		}

		count[tId] = cnt;
	}

	template<typename ContactPair>
	__global__ void CDTS_Narrow_Set(
		DArray<ContactPair> nbr_cons,
		DArray<ContactPair> nbr_cons_all,
		DArray<int> prefix,
		DArray<int> counter)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		int offset = prefix[tId];
		int size = counter[tId];
		for (int n = 0; n < size; n++)
		{
			nbr_cons[offset + n] = nbr_cons_all[8 * tId + n];
		}
	}

	template<typename TDataType>
	void CollistionDetectionTriangleSet<TDataType>::compute()
	{
		//Initialize AABBs of discrete elements
		auto de = this->inDiscreteElements()->constDataPtr();

		int num = de->totalSize();

		if (mQueryAABB.size() != num)
			mQueryAABB.resize(num);

		DArray<Box3D>& boxInGlobal = de->boxesInGlobal();
		DArray<Sphere3D>& sphereInGlobal = de->spheresInGlobal();
		DArray<Tet3D>& tetInGlobal = de->tetsInGlobal();
		DArray<Capsule3D>& capsuleInGlobal = de->capsulesInGlobal();

		ElementOffset elementOffset = de->calculateElementOffset();

		cuExecute(num,
			CDTS_SetupAABBForRigidBodies,
			mQueryAABB,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capsuleInGlobal,
			elementOffset);

		mBroadPhaseCD->inSource()->assign(mQueryAABB);

		auto ts = this->inTriangleSet()->constDataPtr();

		auto& vertices = ts->getPoints();
		auto& indices = ts->getTriangles();

		//Initialize AABBs of the triangle set
		if (this->inTriangleSet()->isModified())
		{
			if (mTriangleAABB.size() != indices.size())
				mTriangleAABB.resize(indices.size());

			cuExecute(indices.size(),
				CDTS_SetupAABBs,
				mTriangleAABB,
				vertices,
				indices);

			mBroadPhaseCD->inTarget()->assign(mTriangleAABB);
		}
		mBroadPhaseCD->update();

		auto& contactList = mBroadPhaseCD->outContactList()->constData();

		mBoundaryContactCounter.resize(num);
		mContactBuffer.resize(8 * num);

		cuExecute(num,
			CDTS_CountContacts,
			mBoundaryContactCounter,
			mContactBuffer,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capsuleInGlobal,
			vertices,
			indices,
			contactList,
			de->shape2RigidBodyMapping(),
			elementOffset);

		mBoundaryContactCpy.assign(mBoundaryContactCounter);

		uint sum = mReduce.accumulate(mBoundaryContactCounter.begin(), mBoundaryContactCounter.size());
		mScan.exclusive(mBoundaryContactCounter, true);

		this->outContacts()->resize(sum);

		cuExecute(num,
			CDTS_Narrow_Set,
			this->outContacts()->getData(),
			mContactBuffer,
			mBoundaryContactCounter,
			mBoundaryContactCpy);
	}

	DEFINE_CLASS(CollistionDetectionTriangleSet);
}