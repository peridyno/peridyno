#include "SimplexSet.h"

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	SimplexSet<TDataType>::SimplexSet()
		: PointSet<TDataType>()
	{
	}

	template<typename TDataType>
	SimplexSet<TDataType>::~SimplexSet()
	{
		mEdgeIndex.clear();
		mTriangleIndex.clear();
		mTetrahedronIndex.clear();
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::copyFrom(SimplexSet<TDataType>& simplex)
	{
		PointSet<TDataType>::copyFrom(simplex);

		mEdgeIndex.assign(simplex.mEdgeIndex);
		mTriangleIndex.assign(simplex.mTriangleIndex);
		mTetrahedronIndex.assign(simplex.mTetrahedronIndex);
	}

	template<typename TDataType>
	bool SimplexSet<TDataType>::isEmpty()
	{
		bool empty = true;
		empty |= mEdgeIndex.size() == 0;
		empty |= mTriangleIndex.size() == 0;
		empty |= mTetrahedronIndex.size() == 0;

		return empty;
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::updateTopology()
	{

	}

	template<typename Edge>
	__global__ void Simplex_SetEdgeIndexIds(
		DArray<uint> ids,
		DArray<Edge> edges)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edges.size()) return;

		Edge edge = edges[tId];

		ids[2 * tId] = edge[0];
		ids[2 * tId + 1] = edge[1];
	}

	__global__ void Simplex_CountVertexNumber(
		DArray<int> counter,
		DArray<uint> vIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vIds.size()) return;

		if (tId == 0 || vIds[tId] != vIds[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Coord>
	__global__ void Simplex_SetupEdgeVertices(
		DArray<Coord> vertices,
		DArray<Coord> originVertices,
		DArray<uint> vertexIdMapper,
		DArray<uint> vIds,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		int shift = radix[tId];
		if (tId == 0 || radix[tId] != radix[tId - 1]) {
			uint vId = vIds[tId];
			vertices[shift] = originVertices[vId];
			vertexIdMapper[vId] = shift;
		}
	}

	template<typename Edge>
	__global__ void Simplex_UpdateEdgeIndex(
		DArray<Edge> newEdgeIndices,
		DArray<Edge> oldEdgeIndices,
		DArray<uint> vertexIdMapper)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= oldEdgeIndices.size()) return;

		Edge edge = oldEdgeIndices[tId];

		uint v0 = vertexIdMapper[edge[0]];
		uint v1 = vertexIdMapper[edge[1]];

		newEdgeIndices[tId] = Edge(v0, v1);
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractSimplex1D(EdgeSet<TDataType>& es)
	{
		es.clear();

		uint eNum = mEdgeIndex.size();

		DArray<uint> vIds(eNum * 2);

		cuExecute(eNum,
			Simplex_SetEdgeIndexIds,
			vIds,
			mEdgeIndex);

		thrust::sort(thrust::device, vIds.begin(), vIds.begin() + vIds.size());

		DArray<int> radix(eNum * 2);

		cuExecute(vIds.size(),
			Simplex_CountVertexNumber,
			radix,
			vIds);

		int vNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Coord> vertices(vNum);
		DArray<Edge> edgeIndices(mEdgeIndex.size());
		DArray<uint> vertexIdMapper(PointSet<TDataType>::mCoords.size());

		cuExecute(radix.size(),
			Simplex_SetupEdgeVertices,
			vertices,
			PointSet<TDataType>::mCoords,
			vertexIdMapper,
			vIds,
			radix);

		cuExecute(eNum,
			Simplex_UpdateEdgeIndex,
			edgeIndices,
			mEdgeIndex,
			vertexIdMapper);

		es.setPoints(vertices);
		es.setEdges(edgeIndices);
		es.update();

		vIds.clear();
		radix.clear();
		vertices.clear();
		vertexIdMapper.clear();
		edgeIndices.clear();
	}

	template<typename Triangle>
	__global__ void Simplex_SetTriangleIndexIds(
		DArray<uint> ids,
		DArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle tri = triangles[tId];

		ids[3 * tId] = tri[0];
		ids[3 * tId + 1] = tri[1];
		ids[3 * tId + 2] = tri[2];
	}

	template<typename Triangle>
	__global__ void Simplex_UpdateTriangleIndex(
		DArray<Triangle> newTriangleIndices,
		DArray<Triangle> oldTriangleIndices,
		DArray<uint> vertexIdMapper)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= oldTriangleIndices.size()) return;

		Triangle tri = oldTriangleIndices[tId];

		uint v0 = vertexIdMapper[tri[0]];
		uint v1 = vertexIdMapper[tri[1]];
		uint v2 = vertexIdMapper[tri[2]];

		newTriangleIndices[tId] = Triangle(v0, v1, v2);
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractSimplex2D(TriangleSet<TDataType>& ts)
	{
		ts.clear();

		uint tNum = mTriangleIndex.size();

		DArray<uint> vIds(tNum * 3);

		cuExecute(tNum,
			Simplex_SetTriangleIndexIds,
			vIds,
			mTriangleIndex);

		thrust::sort(thrust::device, vIds.begin(), vIds.begin() + vIds.size());

		DArray<int> radix(tNum * 3);

		cuExecute(vIds.size(),
			Simplex_CountVertexNumber,
			radix,
			vIds);

		int vNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Coord> vertices(vNum);
		DArray<Triangle> triangleIndex(mTriangleIndex.size());
		DArray<uint> vertexIdMapper(PointSet<TDataType>::mCoords.size());

		cuExecute(radix.size(),
			Simplex_SetupEdgeVertices,
			vertices,
			PointSet<TDataType>::mCoords,
			vertexIdMapper,
			vIds,
			radix);

		cuExecute(tNum,
			Simplex_UpdateTriangleIndex,
			triangleIndex,
			mTriangleIndex,
			vertexIdMapper);

		ts.setPoints(vertices);
		ts.setTriangles(triangleIndex);
		ts.update();

		vIds.clear();
		radix.clear();
		vertices.clear();
		vertexIdMapper.clear();
		triangleIndex.clear();
	}

	template<typename Tetrahedron>
	__global__ void Simplex_SetTetrahedronIndexIds(
		DArray<uint> ids,
		DArray<Tetrahedron> tetrahedrons)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tetrahedrons.size()) return;

		Tetrahedron tet = tetrahedrons[tId];

		ids[4 * tId] = tet[0];
		ids[4 * tId + 1] = tet[1];
		ids[4 * tId + 2] = tet[2];
		ids[4 * tId + 3] = tet[3];
	}

	template<typename Tetrahedron>
	__global__ void Simplex_UpdateTetrahedronIndex(
		DArray<Tetrahedron> newTetIndices,
		DArray<Tetrahedron> oldTetIndices,
		DArray<uint> vertexIdMapper)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= oldTetIndices.size()) return;

		Tetrahedron tet = oldTetIndices[tId];

		uint v0 = vertexIdMapper[tet[0]];
		uint v1 = vertexIdMapper[tet[1]];
		uint v2 = vertexIdMapper[tet[2]];
		uint v3 = vertexIdMapper[tet[3]];

		newTetIndices[tId] = Tetrahedron(v0, v1, v2, v3);
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractSimplex3D(TetrahedronSet<TDataType>& ts)
	{
		ts.clear();

		uint tNum = mTetrahedronIndex.size();

		DArray<uint> vIds(tNum * 4);

		cuExecute(tNum,
			Simplex_SetTetrahedronIndexIds,
			vIds,
			mTetrahedronIndex);

		thrust::sort(thrust::device, vIds.begin(), vIds.begin() + vIds.size());

		DArray<int> radix(tNum * 4);

		cuExecute(vIds.size(),
			Simplex_CountVertexNumber,
			radix,
			vIds);

		int vNum = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Coord> vertices(vNum);
		DArray<Tetrahedron> tetrahedronIndex(mTetrahedronIndex.size());
		DArray<uint> vertexIdMapper(PointSet<TDataType>::mCoords.size());

		cuExecute(radix.size(),
			Simplex_SetupEdgeVertices,
			vertices,
			PointSet<TDataType>::mCoords,
			vertexIdMapper,
			vIds,
			radix);

		cuExecute(tNum,
			Simplex_UpdateTetrahedronIndex,
			tetrahedronIndex,
			mTetrahedronIndex,
			vertexIdMapper);

		ts.setPoints(vertices);
		ts.setTetrahedrons(tetrahedronIndex);
		ts.update();

		vIds.clear();
		radix.clear();
		vertices.clear();
		vertexIdMapper.clear();
		tetrahedronIndex.clear();
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractPointSet(PointSet<TDataType>& ps)
	{
		ps.clear();

		ps.setPoints(PointSet<TDataType>::mCoords);
		ps.update();
	}

	template<typename Edge, typename Triangle, typename Tetrahedron>
	__global__ void Simplex_SetupEdgeKeyAndValue(
		DArray<EKey> keys,
		DArray<Edge> values,
		DArray<Edge> edgeIndices,
		DArray<Triangle> triIndices,
		DArray<Tetrahedron> tetIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		uint total_num = edgeIndices.size() + triIndices.size() + tetIndices.size();

		if (tId >= total_num) return;

		if (tId < edgeIndices.size())
		{
			Edge eIdx = edgeIndices[tId];

			uint v0 = eIdx[0];
			uint v1 = eIdx[1];

			EKey key(v0, v1);

			keys[tId] = key;
			values[tId] = eIdx;
		}
		else if (tId < edgeIndices.size() + triIndices.size())
		{
			uint i = tId - edgeIndices.size();

			Triangle tIdx = triIndices[i];

			uint v0 = tIdx[0];
			uint v1 = tIdx[1];
			uint v2 = tIdx[2];

			uint offset = edgeIndices.size() + 3 * i;

			keys[offset] = EKey(v0, v1);
			values[offset] = Edge(v0, v1);

			keys[offset + 1] = EKey(v1, v2);
			values[offset + 1] = Edge(v1, v2);

			keys[offset + 2] = EKey(v2, v0);
			values[offset + 2] = Edge(v2, v0);
		}
		else
		{
			uint i = tId - edgeIndices.size() - triIndices.size();

			Tetrahedron tIdx = tetIndices[i];

			uint v0 = tIdx[0];
			uint v1 = tIdx[1];
			uint v2 = tIdx[2];
			uint v3 = tIdx[3];

			uint offset = edgeIndices.size() + 3 * triIndices.size() + 6 * i;

			keys[offset] = EKey(v0, v1);
			values[offset] = Edge(v0, v1);

			keys[offset + 1] = EKey(v1, v2);
			values[offset + 1] = Edge(v1, v2);

			keys[offset + 2] = EKey(v2, v0);
			values[offset + 2] = Edge(v2, v0);

			keys[offset + 3] = EKey(v0, v3);
			values[offset + 3] = Edge(v0, v3);

			keys[offset + 4] = EKey(v1, v3);
			values[offset + 4] = Edge(v1, v3);

			keys[offset + 5] = EKey(v2, v3);
			values[offset + 5] = Edge(v2, v3);
		}
	}

	__global__ void Simplex_CountEdgeNumber(
		DArray<int> counter,
		DArray<EKey> vIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vIds.size()) return;

		if (tId == 0 || vIds[tId] != vIds[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Edge>
	__global__ void Simplex_SetupEdges(
		DArray<Edge> newEdges,
		DArray<Edge> oldEdges,
		DArray<EKey> keys,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		int shift = radix[tId];
		if (tId == 0 || radix[tId] != radix[tId - 1]) {
			newEdges[shift] = oldEdges[tId];
		}
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractEdgeSet(EdgeSet<TDataType>& es)
	{
		es.clear();

		uint priNum = mEdgeIndex.size() + mTriangleIndex.size() + mTetrahedronIndex.size();
		uint eNum = mEdgeIndex.size() + 3 * mTriangleIndex.size() + 6 * mTetrahedronIndex.size();

		DArray<EKey> keys(eNum);
		DArray<Edge> edges(eNum);

		cuExecute(priNum,
			Simplex_SetupEdgeKeyAndValue,
			keys,
			edges,
			mEdgeIndex,
			mTriangleIndex,
			mTetrahedronIndex);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), edges.begin());

		DArray<int> radix(eNum);

		cuExecute(eNum,
			Simplex_CountEdgeNumber,
			radix,
			keys);

		int N = thrust::reduce(thrust::device, radix.begin(), radix.begin() + radix.size());
		thrust::exclusive_scan(thrust::device, radix.begin(), radix.begin() + radix.size(), radix.begin());

		DArray<Edge> distinctEdges(N);

		cuExecute(eNum,
			Simplex_SetupEdges,
			distinctEdges,
			edges,
			keys,
			radix);

		es.setPoints(PointSet<TDataType>::mCoords);
		es.setEdges(distinctEdges);
		es.update();

		keys.clear();
		edges.clear();
		radix.clear();
		distinctEdges.clear();
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::extractTriangleSet(TriangleSet<TDataType>& ts)
	{

	}

	DEFINE_CLASS(SimplexSet);
}