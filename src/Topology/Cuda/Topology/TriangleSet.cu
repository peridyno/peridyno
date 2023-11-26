#include "TriangleSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include <thrust/sort.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace dyno
{
	template<typename TDataType>
	TriangleSet<TDataType>::TriangleSet()
		: EdgeSet<TDataType>()
	{
	}

	template<typename TDataType>
	TriangleSet<TDataType>::~TriangleSet()
	{
		mTriangleIndex.clear();
		mVertexNormal.clear();
		mVer2Tri.clear();
		mEdg2Tri.clear();
		mTri2Edg.clear();
	}

	template<typename Triangle>
	__global__ void TS_CountTriangles(
		DArray<uint> counter,
		DArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle t = triangles[tId];

		atomicAdd(&counter[t[0]], 1);
		atomicAdd(&counter[t[1]], 1);
		atomicAdd(&counter[t[2]], 1);
	}

	template<typename Triangle>
	__global__ void TS_SetupTriIds(
		DArrayList<int> triIds,
		DArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle t = triangles[tId];

		triIds[t[0]].atomicInsert(tId);
		triIds[t[1]].atomicInsert(tId);
		triIds[t[2]].atomicInsert(tId);
	}

	template<typename TDataType>
	DArrayList<int>& TriangleSet<TDataType>::getVertex2Triangles()
	{
		DArray<uint> counter(this->mCoords.size());
		counter.reset();

		cuExecute(mTriangleIndex.size(),
			TS_CountTriangles,
			counter,
			mTriangleIndex);

		mVer2Tri.resize(counter);

		counter.reset();
		cuExecute(mTriangleIndex.size(),
			TS_SetupTriIds,
			mVer2Tri,
			mTriangleIndex);

		counter.clear();

		return mVer2Tri;
	}

	template<typename Edg2Tri>
	__global__ void TS_setupIds(
		DArray<int> edgIds,
		DArray<int> triIds,
		DArray<Edg2Tri> edg2Tri)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edg2Tri.size()) return;

		Edg2Tri e = edg2Tri[tId];

		triIds[2 * tId] = e[0];
		triIds[2 * tId + 1] = e[1];

		edgIds[2 * tId] = tId;
		edgIds[2 * tId + 1] = tId;
	}

	template<typename Tri2Edg, typename Edge, typename Triangle>
	__global__ void TS_SetupTri2Edg(
		DArray<Tri2Edg> tri2Edg,
		DArray<int> triIds,
		DArray<int> edgIds,
		DArray<Edge> edges,
		DArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triIds.size()) return;

		if (tId == 0 || triIds[tId] != triIds[tId - 1])
		{
			Tri2Edg t2E(EMPTY, EMPTY, EMPTY);

			EKey te0(triangles[triIds[tId]][0], triangles[triIds[tId]][1]);
			EKey te1(triangles[triIds[tId]][1], triangles[triIds[tId]][2]);
			EKey te2(triangles[triIds[tId]][2], triangles[triIds[tId]][0]);

			EKey e0(edges[edgIds[tId]][0], edges[edgIds[tId]][1]);
			EKey e1(edges[edgIds[tId + 1]][0], edges[edgIds[tId + 1]][1]);
			EKey e2(edges[edgIds[tId + 2]][0], edges[edgIds[tId + 2]][1]);

			if (te0 == e0)
				t2E[0] = edgIds[tId];
			else if (te0 == e1)
				t2E[0] = edgIds[tId + 1];
			else if (te0 == e2)
				t2E[0] = edgIds[tId + 2];

			if (te1 == e0)
				t2E[1] = edgIds[tId];
			else if (te1 == e1)
				t2E[1] = edgIds[tId + 1];
			else if (te1 == e2)
				t2E[1] = edgIds[tId + 2];

			if (te2 == e0)
				t2E[2] = edgIds[tId];
			else if (te2 == e1)
				t2E[2] = edgIds[tId + 1];
			else if (te2 == e2)
				t2E[2] = edgIds[tId + 2];

			int shift = tId / 3;
			tri2Edg[shift] = t2E;

			//printf("tri2Edg: %d, %d %d %d, %d %d %d \n", shift, triangles[triIds[tId]][0], triangles[triIds[tId]][1], triangles[triIds[tId]][2],tri2Edg[shift][0], tri2Edg[shift][1], tri2Edg[shift][2]);
		}
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateTriangle2Edge()
	{
		if (mEdg2Tri.size() == 0)
			this->updateEdges();

		uint edgSize = mEdg2Tri.size();

		DArray<int> triIds, edgIds;
		triIds.resize(2 * edgSize);
		edgIds.resize(2 * edgSize);

		cuExecute(edgSize,
			TS_setupIds,
			edgIds,
			triIds,
			mEdg2Tri);

		thrust::sort_by_key(thrust::device, triIds.begin(), triIds.begin() + triIds.size(), edgIds.begin());

		auto& pEdges = this->getEdges();

		mTri2Edg.resize(mTriangleIndex.size());
		cuExecute(triIds.size(),
			TS_SetupTri2Edg,
			mTri2Edg,
			triIds,
			edgIds,
			pEdges,
			mTriangleIndex);

		triIds.clear();
		edgIds.clear();
	}

	template<typename EKey, typename Triangle>
	__global__ void TS_SetupKeys(
		DArray<EKey> keys,
		DArray<int> ids,
		DArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle tri = triangles[tId];
		keys[3 * tId] = EKey(tri[0], tri[1]);
		keys[3 * tId + 1] = EKey(tri[1], tri[2]);
		keys[3 * tId + 2] = EKey(tri[2], tri[0]);

		ids[3 * tId] = tId;
		ids[3 * tId + 1] = tId;
		ids[3 * tId + 2] = tId;
	}

	template<typename EKey>
	__global__ void TS_CountEdgeNumber(
		DArray<int> counter,
		DArray<EKey> keys)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Edge, typename Edg2Tri, typename EKey>
	__global__ void TS_SetupEdges(
		DArray<Edge> edges,
		DArray<Edg2Tri> edg2Tri,
		DArray<EKey> keys,
		DArray<int> counter,
		DArray<int> triIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int shift = counter[tId];
		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			EKey key = keys[tId];
			edges[shift] = Edge(key[0], key[1]);

			Edg2Tri e2T(EMPTY, EMPTY);
			e2T[0] = triIds[tId];

			if (tId + 1 < keys.size() && keys[tId + 1] == key)
				e2T[1] = triIds[tId + 1];

			edg2Tri[shift] = e2T;
		}
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateEdges()
	{
		uint triSize = mTriangleIndex.size();

		DArray<EKey> keys;
		DArray<int> triIds;

		keys.resize(3 * triSize);
		triIds.resize(3 * triSize);

		cuExecute(triSize,
			TS_SetupKeys,
			keys,
			triIds,
			mTriangleIndex);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), triIds.begin());

		DArray<int> counter;
		counter.resize(3 * triSize);

		cuExecute(keys.size(),
			TS_CountEdgeNumber,
			counter,
			keys);

		int edgeNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		mEdg2Tri.resize(edgeNum);

		auto& pEdges = this->getEdges();
		pEdges.resize(edgeNum);
		cuExecute(keys.size(),
			TS_SetupEdges,
			pEdges,
			mEdg2Tri,
			keys,
			counter,
			triIds);

		counter.clear();
		triIds.clear();
		keys.clear();
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::setTriangles(std::vector<Triangle>& triangles)
	{
		mTriangleIndex.resize(triangles.size());
		mTriangleIndex.assign(triangles);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::setTriangles(DArray<Triangle>& triangles)
	{
		mTriangleIndex.resize(triangles.size());
		mTriangleIndex.assign(triangles);
	}

	template<typename TDataType>
	bool TriangleSet<TDataType>::loadObjFile(std::string filename)
	{
		std::vector<Coord> vertList;
		std::vector<Triangle> faceList;

		tinyobj::attrib_t myattrib;
		std::vector <tinyobj::shape_t> myshape;
		std::vector <tinyobj::material_t> mymat;
		std::string mywarn;
		std::string myerr;

		char* fname = (char*)filename.c_str();

		bool succeed = tinyobj::LoadObj(&myattrib, &myshape, &mymat, &mywarn, &myerr, fname, nullptr ,true, true);
		if (!succeed)
			return false;

		for (int i = 0; i < myattrib.GetVertices().size() / 3; i++)
		{
			vertList.push_back(Coord(myattrib.GetVertices()[3 * i], myattrib.GetVertices()[3 * i + 1], myattrib.GetVertices()[3 * i + 2]));
		}

		for (int i = 0;i < myshape.size();i++) 
		{
			for (int s = 0;s < myshape[i].mesh.indices.size()/3; s++)
			{
				faceList.push_back(Triangle(myshape[i].mesh.indices[3 * s].vertex_index, myshape[i].mesh.indices[3 * s + 1].vertex_index, myshape[i].mesh.indices[3 * s + 2].vertex_index));
			}
		}
		this->setPoints(vertList);
		this->setTriangles(faceList);
		this->update();

		vertList.clear();
		faceList.clear();
		myshape.clear();
		mymat.clear();

		return true;
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::copyFrom(TriangleSet<TDataType>& triangleSet)
	{
		mVer2Tri.assign(triangleSet.mVer2Tri);

		mTriangleIndex.resize(triangleSet.mTriangleIndex.size());
		mTriangleIndex.assign(triangleSet.mTriangleIndex);

		mEdg2Tri.resize(triangleSet.mEdg2Tri.size());
		mEdg2Tri.assign(triangleSet.mEdg2Tri);

		EdgeSet<TDataType>::copyFrom(triangleSet);
	}

	template<typename Triangle>
	__global__ void TS_UpdateIndex(
		DArray<Triangle> indices,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indices.size() - indexOffset) return;

		Triangle t = indices[indexOffset + tId];
		t[0] += vertexOffset;
		t[1] += vertexOffset;
		t[2] += vertexOffset;

		indices[indexOffset + tId] = t;
	}

	template<typename TDataType>
	std::shared_ptr<TriangleSet<TDataType>> TriangleSet<TDataType>::merge(TriangleSet<TDataType>& ts)
	{
		auto ret = std::make_shared<TriangleSet<TDataType>>();

		auto& vertices = ret->getPoints();
		auto& indices = ret->getTriangles();

		uint vNum0 = mCoords.size();
		uint vNum1 = ts.getPoints().size();

		uint tNum0 = mTriangleIndex.size();
		uint tNum1 = ts.getTriangles().size();

		vertices.resize(vNum0 + vNum1);
		indices.resize(tNum0 + tNum1);

		vertices.assign(mCoords, vNum0, 0, 0);
		vertices.assign(ts.getPoints(), vNum1, vNum0, 0);

		indices.assign(mTriangleIndex, tNum0, 0, 0);
		indices.assign(ts.getTriangles(), tNum1, tNum0, 0);

		cuExecute(tNum1,
			TS_UpdateIndex,
			indices,
			vNum0,
			tNum0);

		return ret;
	}

	template<typename TDataType>
	bool TriangleSet<TDataType>::isEmpty()
	{
		return mTriangleIndex.size() == 0 && EdgeSet<TDataType>::isEmpty();
	}

	template<typename Coord, typename Triangle>
	__global__ void TS_SetupVertexNormals(
		DArray<Coord> normals,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArrayList<int> triIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= normals.size()) return;

		List<int>& list_i = triIds[tId];
		int triSize = list_i.size();

		Coord N = Coord(0);
		for (int ne = 0; ne < triSize; ne++)
		{
			int j = list_i[ne];
			Triangle t = triangles[j];

			Coord v0 = vertices[t[0]];
			Coord v1 = vertices[t[1]];
			Coord v2 = vertices[t[2]];

			N += (v1 - v0).cross(v2 - v0);
		}

		N.normalize();

		normals[tId] = N;
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::setNormals(DArray<Coord>& normals)
	{
		mVertexNormal.assign(normals);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateVertexNormal()
	{
		uint vertSize = this->mCoords.size();

		if (vertSize <= 0)
			return;

		if (mVertexNormal.size() != vertSize) {
			mVertexNormal.resize(vertSize);
		}

		auto& vert2Tri = getVertex2Triangles();
		cuExecute(vertSize,
			TS_SetupVertexNormals,
			mVertexNormal,
			this->mCoords,
			mTriangleIndex,
			vert2Tri);
	}
	
	template<typename Coord, typename Triangle>
	__global__ void TS_SetupAngleWeightedVertexNormals(
		DArray<Coord> normals,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArrayList<int> triIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= normals.size()) return;

		List<int>& list_i = triIds[tId];
		int triSize = list_i.size();

		Coord N = Coord(0);
		for (int ne = 0; ne < triSize; ne++)
		{
			int j = list_i[ne];
			Triangle t = triangles[j];

			Coord v0 = vertices[t[0]];
			Coord v1 = vertices[t[1]];
			Coord v2 = vertices[t[2]];

			Real e0 = (v1 - v2).norm();
			Real e1 = (v2 - v0).norm();
			Real e2 = (v1 - v0).norm();

			Real cosangle = 0;
			if (t[0] == tId)
				cosangle = (e1 * e1 + e2 * e2 - e0 * e0) / (2.0 * e1 * e2);
			else if (t[1] == tId)
				cosangle = (e0 * e0 + e2 * e2 - e1 * e1) / (2.0 * e0 * e2);
			else if (t[2] == tId)
				cosangle = (e1 * e1 + e0 * e0 - e2 * e2) / (2.0 * e1 * e0);

			Real angle = acos(cosangle);
			Coord norm = (v1 - v0).cross(v2 - v0);
			norm.normalize();
			N += angle * norm;

			//printf("vertex normal: %d, %f %f %f, %d %f, %f %f %f, %f %f %f \n", tId, vertices[tId][0], vertices[tId][1], vertices[tId][2],
			//	j, angle, norm[0], norm[1], norm[2], N[0], N[1], N[2]);
		}

		N.normalize();

		normals[tId] = N;
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateAngleWeightedVertexNormal(DArray<Coord>& vertexNormal)
	{
		uint vertSize = this->mCoords.size();

		vertexNormal.resize(vertSize);

		auto& vert2Tri = getVertex2Triangles();

		cuExecute(vertSize,
			TS_SetupAngleWeightedVertexNormals,
			vertexNormal,
			this->mCoords,
			mTriangleIndex,
			vert2Tri);
	}

	template<typename Coord, typename Triangle, typename Edg2Tri>
	__global__ void TS_SetupEdgeNormals(
		DArray<Coord> normals,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<Edg2Tri> triIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= normals.size()) return;

		Edg2Tri& edge = triIds[tId];

		Coord N = Coord(0);
		for (int ne = 0; ne < 2; ne++)
		{
			int j = edge[ne];
			Triangle t = triangles[j];

			Coord v0 = vertices[t[0]];
			Coord v1 = vertices[t[1]];
			Coord v2 = vertices[t[2]];

			Coord norm = (v1 - v0).cross(v2 - v0);
			norm.normalize();
			N += norm;			
		}

		N.normalize();

		normals[tId] = N;
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateEdgeNormal(DArray<Coord>& edgeNormal)
	{
		if (mEdg2Tri.size() == 0)
			updateEdges();

		edgeNormal.resize(mEdg2Tri.size());

		cuExecute(mEdg2Tri.size(),
			TS_SetupEdgeNormals,
			edgeNormal,
			this->mCoords,
			mTriangleIndex,
			mEdg2Tri);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateTopology()
	{
		this->updateTriangles();

		if(bAutoUpdateNormal)
			this->updateVertexNormal();

		this->EdgeSet<TDataType>::updateTopology();
	}

	template <typename Real, typename Coord>
	__global__ void PS_RotateNormal(
		DArray<Coord> normals,
		Quat<Real> q)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= normals.size()) return;
		SquareMatrix<Real, 3> rot = q.toMatrix3x3();

		normals[pId] = rot * normals[pId];
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::rotate(const Coord angle)
	{
		EdgeSet<TDataType>::rotate(angle);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::rotate(const Quat<Real> q)
	{
		EdgeSet<TDataType>::rotate(q);

		cuExecute(mVertexNormal.size(), PS_RotateNormal, mVertexNormal, q);
	}

	DEFINE_CLASS(TriangleSet);
}