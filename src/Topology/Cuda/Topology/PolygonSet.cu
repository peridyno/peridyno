#include "PolygonSet.h"

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	PolygonSet<TDataType>::PolygonSet()
		: EdgeSet<TDataType>()
	{
	}

	template<typename TDataType>
	PolygonSet<TDataType>::~PolygonSet()
	{
		mPolygonIndex.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::setPolygons(const CArrayList<uint>& indices)
	{
		mPolygonIndex.assign(indices);
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::setPolygons(const DArrayList<uint>& indices)
	{
		mPolygonIndex.assign(indices);
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

	__global__ void PolygonSet_CountPolygonNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= polygonIndices.size()) return;

		auto& index = polygonIndices[tId];

		int N = index.size();
		for (int i = 0; i < N; i++)
		{
			uint v0 = index[i];
			atomicAdd(&counter[v0], 1);
		}
	}

	__global__ void PolygonSet_SetupVertex2Polygon(
		DArrayList<uint> vert2Poly,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= polygonIndices.size()) return;

		auto& index = polygonIndices[tId];

		int N = index.size();
		for (int i = 0; i < N; i++)
		{
			uint v0 = index[i];
			index.atomicInsert(v0);
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::updateTopology()
	{
		uint vNum = PointSet<TDataType>::mCoords.size();

		//Update the vertex to polygon mapping
		DArray<uint> counter(vNum);
		counter.reset();

		cuExecute(vNum,
			PolygonSet_CountPolygonNumber,
			counter,
			mPolygonIndex);

		mVer2Poly.resize(counter);

		cuExecute(vNum,
			PolygonSet_SetupVertex2Polygon,
			mVer2Poly,
			mPolygonIndex);

		counter.clear();

		EdgeSet<TDataType>::updateTopology();
	}

	__global__ void PolygonSet_CountEdgeNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size();
	}

	__global__ void PolygonSet_SetupEdgeKeys(
		DArray<EKey> keys,
		DArray<uint> polyIds,
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
			keys[offset + i] = EKey(v0, v1);
			polyIds[offset + i] = tId;
		}
	}

	__global__ void PolygonSet_CountUniqueEdge(
		DArray<uint> counter,
		DArray<EKey> keys)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Edge, typename Edg2Poly>
	__global__ void PolygonSet_SetupEdgeIndices(
		DArray<Edge> edges,
		DArrayList<uint> poly2Edge,
		DArray<Edg2Poly> edg2Poly,
		DArray<EKey> edgeKeys,
		DArray<uint> polyIds,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edgeKeys.size()) return;

		uint edgeId = radix[tId];
		uint polyId = polyIds[tId];
		poly2Edge[polyId].atomicInsert(edgeId);
		if (tId == 0 || edgeKeys[tId] != edgeKeys[tId - 1])
		{
			EKey key = edgeKeys[tId];
			edges[edgeId] = Edge(key[0], key[1]);

			Edg2Poly e2p(EMPTY, EMPTY);
			e2p[0] = polyIds[tId];

			if (tId + 1 < edgeKeys.size() && edgeKeys[tId + 1] == key)
				e2p[1] = polyIds[tId + 1];

			edg2Poly[edgeId] = e2p;
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::updateEdges()
	{
		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		cuExecute(polyNum,
			PolygonSet_CountEdgeNumber,
			radix,
			mPolygonIndex);

		mPoly2Edg.resize(radix);

		int eNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<EKey> edgeKeys(eNum);
		DArray<uint> polyIds(eNum);

		cuExecute(polyNum,
			PolygonSet_SetupEdgeKeys,
			edgeKeys,
			polyIds,
			mPolygonIndex,
			radix);

		DArray<uint> uniqueEdgeCounter(eNum);

		thrust::sort_by_key(thrust::device, edgeKeys.begin(), edgeKeys.begin() + edgeKeys.size(), polyIds.begin());

		cuExecute(edgeKeys.size(),
			PolygonSet_CountUniqueEdge,
			uniqueEdgeCounter,
			edgeKeys);

		int uniqueNum = thrust::reduce(thrust::device, uniqueEdgeCounter.begin(), uniqueEdgeCounter.begin() + uniqueEdgeCounter.size());
		thrust::exclusive_scan(thrust::device, uniqueEdgeCounter.begin(), uniqueEdgeCounter.begin() + uniqueEdgeCounter.size(), uniqueEdgeCounter.begin());

		EdgeSet<TDataType>::mEdges.resize(uniqueNum);
		mEdg2Poly.resize(uniqueNum);

		cuExecute(edgeKeys.size(),
			PolygonSet_SetupEdgeIndices,
			EdgeSet<TDataType>::mEdges,
			mPoly2Edg,
			mEdg2Poly,
			edgeKeys,
			polyIds,
			uniqueEdgeCounter);

		radix.clear();
		polyIds.clear();
		edgeKeys.clear();
		uniqueEdgeCounter.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractEdgeSet(EdgeSet<TDataType>& es)
	{
		es.setPoints(PointSet<TDataType>::mCoords);
		es.setEdges(EdgeSet<TDataType>::mEdges);

		es.update();
	}

	__global__ void PolygonSet_ExtractTriangleNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size() == 3 ? 1 : 0;
	}

	template<typename Triangle>
	__global__ void PolygonSet_ExtractTriangleIndices(
		DArray<Triangle> triangles,
		DArrayList<uint> polygonIndices,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		uint offset = radix[tId];

		auto& index = polygonIndices[tId];

		if (index.size() == 3)
		{
			uint v0 = index[0];
			uint v1 = index[1];
			uint v2 = index[2];
			triangles[offset] = Triangle(v0, v1, v2);
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractTriangleSet(TriangleSet<TDataType>& ts)
	{
		ts.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		cuExecute(polyNum,
			PolygonSet_ExtractTriangleNumber,
			radix,
			mPolygonIndex);

		int tNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Triangle> triangleIndices(tNum);

		//TODO: remove duplicative vertices
		cuExecute(polyNum,
			PolygonSet_ExtractTriangleIndices,
			triangleIndices,
			mPolygonIndex,
			radix);

		ts.setPoints(PointSet<TDataType>::mCoords);
		ts.setTriangles(triangleIndices);
		ts.update();

		radix.clear();
		triangleIndices.clear();
	}

	__global__ void PolygonSet_ExtractQuadNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size() == 4 ? 1 : 0;
	}

	template<typename Quad>
	__global__ void PolygonSet_ExtractQuadIndices(
		DArray<Quad> quads,
		DArrayList<uint> polygonIndices,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		uint offset = radix[tId];

		auto& index = polygonIndices[tId];

		if (index.size() == 4)
		{
			uint v0 = index[0];
			uint v1 = index[1];
			uint v2 = index[2];
			uint v3 = index[3];
			quads[offset] = Quad(v0, v1, v2, v3);
		}
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractQuadSet(QuadSet<TDataType>& qs)
	{
		qs.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		cuExecute(polyNum,
			PolygonSet_ExtractQuadNumber,
			radix,
			mPolygonIndex);

		int tNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Quad> quadIndices(tNum);

		//TODO: remove duplicative vertices
		cuExecute(polyNum,
			PolygonSet_ExtractQuadIndices,
			quadIndices,
			mPolygonIndex,
			radix);

		qs.setPoints(PointSet<TDataType>::mCoords);
		qs.setQuads(quadIndices);
		qs.update();

		radix.clear();
		quadIndices.clear();
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
	void PolygonSet<TDataType>::turnIntoTriangleSet(TriangleSet<TDataType>& ts)
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

		ts.setPoints(PointSet<TDataType>::mCoords);
		ts.setTriangles(triangleIndex);
		ts.update();

		radix.clear();
		triangleIndex.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::triangleSetToPolygonSet(TriangleSet<TDataType>& ts) 
	{
		this->setPoints(ts.getPoints());
		
		this->mPolygonIndex.resize(ts.getTriangles().size(),3);

		cuExecute(ts.getTriangles().size(),
			triSet2PolygonSet,
			ts.getTriangles(),
			this->mPolygonIndex
		);

		this->update();

	}


	template<typename Triangle>
	__global__ void triSet2PolygonSet(
		DArray<Triangle> triangles,
		DArrayList<uint> polygons
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		auto& index = polygons[tId];
		for (int i = 0; i < 3; i++)
			index.insert(triangles[tId][i]);
		
	}

	DEFINE_CLASS(PolygonSet);
}