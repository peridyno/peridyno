#pragma once
#include <cuda_runtime.h>
#include "TriangleSetToTriangleSet.h"
#include "Topology/NeighborQuery.h"
#include "Algorithm/MatrixFunc.h"
#include "Utility.h"
#include "Topology/Primitive3D.h"
#include "Collision/CollisionDetectionBroadPhase.h"

namespace dyno
{
	typedef typename TAlignedBox3D<Real> AABB;

	template<typename TDataType>
	TriangleSetToTriangleSet<TDataType>::TriangleSetToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	TriangleSetToTriangleSet<TDataType>::TriangleSetToTriangleSet(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<TriangleSet<TDataType>> to)
	{
		m_from = from;
		m_to = to;
	}

	template<typename TDataType>
	TriangleSetToTriangleSet<TDataType>::~TriangleSetToTriangleSet()
	{
		m_nearestTriangle.release();
		m_barycentric.release();
	}

	template<typename TDataType>
	bool TriangleSetToTriangleSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

	template<typename Real, typename Coord>
	__device__ Coord CalculateLocalCoord(
		Coord worldLoc,
		TTriangle3D<Real> triangle)
	{
// 		Coord e0 = triangle.v[1] - triangle.v[0];
// 		Coord e1 = triangle.v[2] - triangle.v[0];
// 		e0.normalize();
// 		Coord e2 = e0.cross(e1);
// 		e2.normalize();
// 		e1 = e2.cross(e0);
// 
// 		Coord dir = worldLoc - triangle.v[0];
// 
// 		return Coord(dir.dot(e0), dir.dot(e1), dir.dot(e2));

		typename TTriangle3D<Real>::Param bary;
		triangle.computeBarycentrics(worldLoc, bary);

		return Coord(bary.u, bary.v, bary.w);
	}

	template<typename Real, typename Coord>
	__device__ Coord CalculateWorldCoord(
		Coord localLoc,
		TTriangle3D<Real> triangle)
	{
// 		Coord e0 = triangle.v[1] - triangle.v[0];
// 		Coord e1 = triangle.v[2] - triangle.v[0];
// 		e0.normalize();
// 		Coord e2 = e0.cross(e1);
// 		e2.normalize();
// 		e1 = e2.cross(e0);
// 
// 		return triangle.v[0] + localLoc[0] * e0 + localLoc[1] * e1 + localLoc[2] * e2;

		typename TTriangle3D<Real>::Param param;
		param.u = localLoc[0];
		param.v = localLoc[1];
		param.w = localLoc[2];

		return triangle.computeLocation(param);
	}

	template <typename Coord, typename Triangle>
	__global__ void K_ApplyTransform(
		GArray<Coord> tarVertex,
		GArray<Coord> tarInitVertex,
		GArray<Coord> srcVertex,
		GArray<Coord> srcInitVertex,
		GArray<Triangle> srcTriangle,
		GArray<int> nearestTriangleIndex,
		GArray<Coord> barycentric)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= tarVertex.size()) return;

		Coord b = barycentric[vId];

		Triangle tIndex = srcTriangle[nearestTriangleIndex[vId]];
		Coord dv0 = srcVertex[tIndex[0]] - srcInitVertex[tIndex[0]];
		Coord dv1 = srcVertex[tIndex[1]] - srcInitVertex[tIndex[1]];
		Coord dv2 = srcVertex[tIndex[2]] - srcInitVertex[tIndex[2]];

		Coord dv = b[0] * dv0 + b[1] * dv1 + b[2] * dv2;

		tarVertex[vId] = tarInitVertex[vId] + dv;
	}

	template<typename TDataType>
	bool TriangleSetToTriangleSet<TDataType>::apply()
	{
		cuExecute(m_to->getPoints().size(),
			K_ApplyTransform,
			m_to->getPoints(),
			m_initTo->getPoints(),
			m_from->getPoints(),
			m_initFrom->getPoints(),
			*(m_from->getTriangles()),
			m_nearestTriangle,
			m_barycentric);

		return true;
	}

	template <typename Coord, typename Triangle>
	__global__ void SetupBoundBoxFrom(
		GArray<AABB> boundingBox,
		GArray<Coord> points,
		GArray<Triangle> triangleIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangleIndex.size()) return;

		Triangle index = triangleIndex[tId];
		TTriangle3D<Real> tri3d(points[index[0]], points[index[1]], points[index[2]]);

		boundingBox[tId] = tri3d.aabb();
	}

	template <typename Real, typename Coord>
	__global__ void SetupBoundBoxTo(
		GArray<AABB> boundingBox,
		GArray<Coord> points,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= boundingBox.size()) return;

		Coord center = points[pId];

		boundingBox[pId] = AABB(center - radius, center + radius);
	}

	template <typename Coord, typename Triangle>
	__global__ void SetupInternalData(
		GArray<int> nearestIds,
		GArray<Coord> barycentric,
		GArray<Coord> toVertex,
		GArray<Coord> fromVertex,
		GArray<Triangle> fromTriangle,
		GArray<TopologyModule::Tri2Tet> tri2Tet,
		NeighborList<int> neighborIds)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= nearestIds.size()) return;

		Coord tarPoint = toVertex[vId];

		int minId = 0;
		Real minDist = Real(10000);

		int nbSize = neighborIds.getNeighborSize(vId);

// 		if (nbSize <= 0)
// 		{
// 			printf("%f %f %f \n", tarPoint[0], tarPoint[1], tarPoint[2]);
// 		}

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighborIds.getElement(vId, ne);

			Triangle tIndex = fromTriangle[j];
			TopologyModule::Tri2Tet t2t = tri2Tet[j];

			TTriangle3D<Real> t(fromVertex[tIndex[0]], fromVertex[tIndex[1]], fromVertex[tIndex[2]]);
			TPoint3D<Real> p(tarPoint);

			Real dist = p.distanceSquared(t);

			if (dist < minDist && (t2t[0] == EMPTY || t2t[1] == EMPTY))
			{
				minDist = dist;
				minId = j;
			}
		}

		Triangle tIndex = fromTriangle[minId];
		TTriangle3D<Real> nearestT(fromVertex[tIndex[0]], fromVertex[tIndex[1]], fromVertex[tIndex[2]]);

		barycentric[vId] = CalculateLocalCoord(tarPoint, nearestT);
		nearestIds[vId] = minId;
	}

	template<typename TDataType>
	void TriangleSetToTriangleSet<TDataType>::match(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<TriangleSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<TetrahedronSet<TDataType>>();
		m_initTo = std::make_shared<TriangleSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);

		int fromTriangleSize = from->getTriangles()->size();
		int toVertSize = to->getPoints().size();

		GArray<AABB> fromAABB;
		GArray<AABB> toAABB;
		
		fromAABB.resize(fromTriangleSize);
		toAABB.resize(toVertSize);

		cuExecute(fromTriangleSize,
			SetupBoundBoxFrom,
			fromAABB,
			from->getPoints(),
			*(from->getTriangles()));

		cuExecute(toVertSize,
			SetupBoundBoxTo,
			toAABB,
			to->getPoints(),
			m_radius);

		auto broadPhase = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
		broadPhase->inTarget()->setValue(fromAABB);
		broadPhase->inSource()->setValue(toAABB);

		broadPhase->update();

		m_nearestTriangle.resize(toVertSize);
		m_barycentric.resize(toVertSize);

		cuExecute(toVertSize,
			SetupInternalData,
			m_nearestTriangle,
			m_barycentric,
			to->getPoints(),
			from->getPoints(),
			*(from->getTriangles()),
			from->getTri2Tet(),
			broadPhase->outContactList()->getValue());

		fromAABB.release();
		toAABB.release();
	}
}