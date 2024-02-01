#include "PolygonSet.h"

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	PolygonSet<TDataType>::PolygonSet()
		: PointSet<TDataType>()
	{
	}

	template<typename TDataType>
	PolygonSet<TDataType>::~PolygonSet()
	{
		mPolygonIndex.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::copyFrom(PolygonSet<TDataType>& polygons)
	{
		PointSet<TDataType>::copyFrom(polygons);

		mPolygonIndex.assign(polygons.mPolygonIndex);
	}

	template<typename TDataType>
	bool PolygonSet<TDataType>::isEmpty()
	{
		bool empty = true;
		empty |= mPolygonIndex.size() == 0;

		return empty;
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::updateTopology()
	{

	}

	__global__ void PolygonSet_CountEdgeNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size();
	}

	template<typename Edge>
	__global__ void PolygonSet_SetupEdgeIndices(
		DArray<Edge> edges,
		DArrayList<uint> polygonIndices,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		uint offset = radix[tId];

		auto& index = polygonIndices[tId];

		int N = index.size();
		for (int i = 0; i < N; i++)
		{
			uint v0 = index[i];
			uint v1 = index[(i + 1) % N];
			edges[offset + i] = Edge(v0, v1);
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractEdgeSet(EdgeSet<TDataType>& es)
	{
		es.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		cuExecute(polyNum,
			PolygonSet_CountEdgeNumber,
			radix,
			mPolygonIndex);

		int eNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Edge> edges(eNum);

		//TODO: remove duplicates
		cuExecute(polyNum,
			PolygonSet_SetupEdgeIndices,
			edges,
			mPolygonIndex,
			radix);

		es.setPoints(mCoords);
		es.setEdges(edges);
		es.update();

		radix.clear();
		edges.clear();
	}

	__global__ void PolygonSet_CountTriangleNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size() - 2;
	}

	template<typename Triangle>
	__global__ void PolygonSet_SetupTriangleIndices(
		DArray<Triangle> triangles,
		DArrayList<uint> polygonIndices,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		uint offset = radix[tId];

		auto& index = polygonIndices[tId];

		int N = index.size();
		for (int i = 0; i < N - 2; i++)
		{
			uint v0 = index[0];
			uint v1 = index[i + 1];
			uint v2 = index[i + 2];
			triangles[offset + i] = Triangle(v0, v1, v2);
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractTriangleSet(TriangleSet<TDataType>& ts)
	{
		ts.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		cuExecute(polyNum,
			PolygonSet_CountTriangleNumber,
			radix,
			mPolygonIndex);

		int tNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Triangle> triangleIndex(tNum);

		cuExecute(polyNum,
			PolygonSet_SetupTriangleIndices,
			triangleIndex,
			mPolygonIndex,
			radix);

		ts.setPoints(mCoords);
		ts.setTriangles(triangleIndex);
		ts.update();

		radix.clear();
		triangleIndex.clear();
	}

	DEFINE_CLASS(PolygonSet);
}