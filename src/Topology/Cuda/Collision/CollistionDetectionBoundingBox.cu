#include "CollistionDetectionBoundingBox.h"

#include "Primitive/Primitive3D.h"
#include "Topology/DiscreteElements.h"

namespace dyno
{
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	CollistionDetectionBoundingBox<TDataType>::CollistionDetectionBoundingBox()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	CollistionDetectionBoundingBox<TDataType>::~CollistionDetectionBoundingBox()
	{

	}

	template <typename Coord>
	__global__ void CountContactsWithBoundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<Tet3D> tet,
		DArray<Capsule3D> cap,
		DArray<int> count,
		Coord hi,
		Coord lo,
		ElementOffset elementOffset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= count.size()) return;

		ElementType eleType = elementOffset.checkElementType(pId);

		if (eleType == ET_SPHERE)//sphere
		{
			int cnt = 0;

			Sphere3D sp = sphere[pId - elementOffset.sphereIndex()];

			Real radius = sp.radius;
			Coord center = sp.center;

			if (center.x + radius >= hi.x)
			{
				cnt++;
			}
			if (center.x - radius <= lo.x)
			{
				cnt++;
			}

			if (center.y + radius >= hi.y)
			{
				cnt++;
			}
			if (center.y - radius <= lo.y)
			{
				cnt++;
			}

			if (center.z + radius >= hi.z)
			{
				cnt++;
			}
			if (center.z - radius <= lo.z)
			{
				cnt++;
			}

			count[pId] = cnt;
		}
		else if (eleType == ET_BOX)//box
		{
			//int idx = pId - start_box;
			int cnt = 0;
			//				int start_i;
			Coord center = box[pId - elementOffset.boxIndex()].center;
			Coord u = box[pId - elementOffset.boxIndex()].u;
			Coord v = box[pId - elementOffset.boxIndex()].v;
			Coord w = box[pId - elementOffset.boxIndex()].w;
			Coord extent = box[pId - elementOffset.boxIndex()].extent;
			Point3D p[8];
			p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
			p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
			p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
			p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
			p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
			p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
			p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
			p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
			bool c1, c2, c3, c4, c5, c6;
			c1 = c2 = c3 = c4 = c5 = c6 = true;
			for (int i = 0; i < 8; i++)
			{
				Coord pos = p[i].origin;
				if (pos[0] > hi[0] && c1)
				{
					c1 = true;
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					cnt++;
				}
			}
			count[pId] = cnt;
		}
		else if (eleType == ET_TET) // tets
		{
			int cnt = 0;
			int start_i = count[pId];

			Tet3D tet_i = tet[pId - elementOffset.tetIndex()];

			for (int i = 0; i < 4; i++)
			{
				Coord vertex = tet_i.v[i];
				if (vertex.x >= hi.x)
				{
					cnt++;
				}
				if (vertex.x <= lo.x)
				{
					cnt++;
				}

				if (vertex.y >= hi.y)
				{
					cnt++;
				}
				if (vertex.y <= lo.y)
				{
					cnt++;
				}

				if (vertex.z >= hi.z)
				{
					cnt++;
				}
				if (vertex.z <= lo.z)
				{
					cnt++;
				}
			}

			count[pId] = cnt;
		}
		else if (eleType == ET_CAPSULE)//segments
		{
			int cnt = 0;

			Capsule3D cap_i = cap[pId - elementOffset.capsuleIndex()];

			Coord v0 = cap_i.startPoint();
			Coord v1 = cap_i.endPoint();

			Real radius = cap_i.radius;

			if (v0.x + radius >= hi.x)
			{
				cnt++;
			}
			if (v0.x - radius <= lo.x)
			{
				cnt++;
			}

			if (v0.y + radius >= hi.y)
			{
				cnt++;
			}
			if (v0.y - radius <= lo.y)
			{
				cnt++;
			}

			if (v0.z + radius >= hi.z)
			{
				cnt++;
			}
			if (v0.z - radius <= lo.z)
			{
				cnt++;
			}


			//v1
			if (v1.x + radius >= hi.x)
			{
				cnt++;
			}
			if (v1.x - radius <= lo.x)
			{
				cnt++;
			}

			if (v1.y + radius >= hi.y)
			{
				cnt++;
			}
			if (v1.y - radius <= lo.y)
			{
				cnt++;
			}

			if (v1.z + radius >= hi.z)
			{
				cnt++;
			}
			if (v1.z - radius <= lo.z)
			{
				cnt++;
			}

			count[pId] = cnt;
		}
	}

	template <typename Coord, typename ContactPair>
	__global__ void SetupContactsWithBoundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<Tet3D> tet,
		DArray<Capsule3D> cap,
		DArray<int> count,
		DArray<ContactPair> nbq,
		DArray<Pair<uint, uint>> mapping,
		Coord hi,
		Coord lo,
		ElementOffset elementOffset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= count.size()) return;

		ElementType eleType = elementOffset.checkElementType(pId);

		uint rbId = mapping[pId].second;

		if (eleType == ET_SPHERE)//sphere
		{
			int cnt = 0;
			int start_i = count[pId];

			Sphere3D sp = sphere[pId - elementOffset.sphereIndex()];

			Real radius = sp.radius;
			Coord center = sp.center;

			if (center.x + radius >= hi.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
				nbq[cnt + start_i].pos1 = center + Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.x + radius - hi.x;
				cnt++;
			}
			if (center.x - radius <= lo.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
				nbq[cnt + start_i].pos1 = center - Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.x - (center.x - radius);
				cnt++;
			}

			if (center.y + radius >= hi.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
				nbq[cnt + start_i].pos1 = center + Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.y + radius - hi.y;
				cnt++;
			}
			if (center.y - radius <= lo.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
				nbq[cnt + start_i].pos1 = center - Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.y - (center.y - radius);
				cnt++;
			}

			if (center.z + radius >= hi.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
				nbq[cnt + start_i].pos1 = center + Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.z + radius - hi.z;
				cnt++;
			}
			if (center.z - radius <= lo.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
				nbq[cnt + start_i].pos1 = center - Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.z - (center.z - radius);
				cnt++;
			}
		}
		else if (eleType == ET_BOX)//box
		{
			//int idx = pId - start_box;
			int cnt = 0;
			int start_i = count[pId];
			Coord center = box[pId - elementOffset.boxIndex()].center;
			Coord u = box[pId - elementOffset.boxIndex()].u;
			Coord v = box[pId - elementOffset.boxIndex()].v;
			Coord w = box[pId - elementOffset.boxIndex()].w;
			Coord extent = box[pId - elementOffset.boxIndex()].extent;
			Point3D p[8];
			p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
			p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
			p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
			p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
			p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
			p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
			p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
			p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
			bool c1, c2, c3, c4, c5, c6;
			c1 = c2 = c3 = c4 = c5 = c6 = true;
			for (int i = 0; i < 8; i++)
			{
				Coord pos = p[i].origin;
				if (pos[0] > hi[0] && c1)
				{
					c1 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[0] - hi[0];
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[1] - hi[1];
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[2] - hi[2];
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[0] - pos[0];
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[1] - pos[1];
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[2] - pos[2];
					cnt++;
				}

			}
		}
		else if (eleType == ET_TET) // tets
		{
			int cnt = 0;
			int start_i = count[pId];

			Tet3D tet_i = tet[pId - elementOffset.tetIndex()];

			for (int i = 0; i < 4; i++)
			{
				Coord vertex = tet_i.v[i];
				if (vertex.x >= hi.x)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.x - hi.x;
					cnt++;
				}
				if (vertex.x <= lo.x)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.x - (vertex.x);
					cnt++;
				}

				if (vertex.y >= hi.y)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.y - hi.y;
					cnt++;
				}
				if (vertex.y <= lo.y)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.y - (vertex.y);
					cnt++;
				}

				if (vertex.z >= hi.z)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.z - hi.z;
					cnt++;
				}
				if (vertex.z <= lo.z)
				{
					nbq[cnt + start_i].bodyId1 = rbId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.z - (vertex.z);
					cnt++;
				}
			}
		}
		else if (eleType == ET_CAPSULE)
		{
			int cnt = 0;
			int start_i = count[pId];

			Capsule3D cap_i = cap[pId - elementOffset.capsuleIndex()];

			Coord v0 = cap_i.startPoint();
			Coord v1 = cap_i.endPoint();

			Real radius = cap_i.radius;

			//v0
			if (v0.x + radius >= hi.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
				nbq[cnt + start_i].pos1 = v0 + Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v0.x + radius - hi.x;
				cnt++;
			}
			if (v0.x - radius <= lo.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
				nbq[cnt + start_i].pos1 = v0 - Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.x - (v0.x - radius);
				cnt++;
			}

			if (v0.y + radius >= hi.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
				nbq[cnt + start_i].pos1 = v0 + Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v0.y + radius - hi.y;
				cnt++;
			}
			if (v0.y - radius <= lo.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
				nbq[cnt + start_i].pos1 = v0 - Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.y - (v0.y - radius);
				cnt++;
			}

			if (v0.z + radius >= hi.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
				nbq[cnt + start_i].pos1 = v0 + Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v0.z + radius - hi.z;
				cnt++;
			}
			if (v0.z - radius <= lo.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
				nbq[cnt + start_i].pos1 = v0 - Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.z - (v0.z - radius);
				cnt++;
			}

			//v1
			if (v1.x + radius >= hi.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
				nbq[cnt + start_i].pos1 = v1 + Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v1.x + radius - hi.x;
				cnt++;
			}
			if (v1.x - radius <= lo.x)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
				nbq[cnt + start_i].pos1 = v1 - Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.x - (v1.x - radius);
				cnt++;
			}

			if (v1.y + radius >= hi.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
				nbq[cnt + start_i].pos1 = v1 + Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v1.y + radius - hi.y;
				cnt++;
			}
			if (v1.y - radius <= lo.y)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
				nbq[cnt + start_i].pos1 = v1 - Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.y - (v1.y - radius);
				cnt++;
			}

			if (v1.z + radius >= hi.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
				nbq[cnt + start_i].pos1 = v1 + Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = v1.z + radius - hi.z;
				cnt++;
			}
			if (v1.z - radius <= lo.z)
			{
				nbq[cnt + start_i].bodyId1 = rbId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
				nbq[cnt + start_i].pos1 = v1 - Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.z - (v1.z - radius);
				cnt++;
			}
		}
	}

	template<typename TDataType>
	void CollistionDetectionBoundingBox<TDataType>::compute()
	{
		int sum = 0;

		auto upperBound = this->varUpperBound()->getData();
		auto lowerBound = this->varLowerBound()->getData();

		auto discreteSet = this->inDiscreteElements()->getDataPtr();
		uint totalSize = discreteSet->totalSize();

		DArray<Box3D> boxInGlobal;
		DArray<Sphere3D> sphereInGlobal;
		DArray<Tet3D> tetInGlobal;
		DArray<Capsule3D> capsuleInGlobal;

		discreteSet->requestDiscreteElementsInGlobal(boxInGlobal, sphereInGlobal, tetInGlobal, capsuleInGlobal);

		ElementOffset offset = discreteSet->calculateElementOffset();

		mBoundaryContactCounter.resize(discreteSet->totalSize());
		mBoundaryContactCounter.reset();
		if (discreteSet->totalSize() > 0)
		{
			cuExecute(totalSize,
				CountContactsWithBoundary,
				sphereInGlobal,
				boxInGlobal,
				tetInGlobal,
				capsuleInGlobal,
				mBoundaryContactCounter,
				upperBound,
				lowerBound,
				offset);

			sum += mReduce.accumulate(mBoundaryContactCounter.begin(), mBoundaryContactCounter.size());
			mScan.exclusive(mBoundaryContactCounter, true);

			this->outContacts()->resize(sum);

			if (sum > 0) {
				cuExecute(totalSize,
					SetupContactsWithBoundary,
					sphereInGlobal,
					boxInGlobal,
					tetInGlobal,
					capsuleInGlobal,
					mBoundaryContactCounter,
					this->outContacts()->getData(),
					discreteSet->shape2RigidBodyMapping(),
					upperBound,
					lowerBound,
					offset);
			}
		}
		else
			this->outContacts()->resize(0);

		boxInGlobal.clear();
		sphereInGlobal.clear();
		tetInGlobal.clear();
		capsuleInGlobal.clear();
	}

	DEFINE_CLASS(CollistionDetectionBoundingBox);
}