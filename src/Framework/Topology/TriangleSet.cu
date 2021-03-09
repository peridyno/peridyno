#include "TriangleSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	TriangleSet<TDataType>::TriangleSet()
		: EdgeSet<TDataType>()
	{
		std::vector<Coord> positions;
		std::vector<Triangle> triangles;
		float dx = 0.1;
		int Nx = 11;
		int Nz = 11;

		for (int k = 0; k < Nz; k++) {
			for (int i = 0; i < Nx; i++) {
				positions.push_back(Coord(Real(i*dx), Real(0.0), Real(k*dx)));
				if (k < Nz - 1 && i < Nx - 1)
				{
					Triangle tri1(i + k*Nx, i + 1 + k*Nx, i + 1 + (k + 1)*Nx);
					Triangle tri2(i + k*Nx, i + 1 + (k + 1)*Nx, i + (k + 1)*Nx);
					triangles.push_back(tri1);
					triangles.push_back(tri2);
				}
			}
		}
		this->setPoints(positions);
		this->setTriangles(triangles);
	}

	template<typename TDataType>
	TriangleSet<TDataType>::~TriangleSet()
	{
	}

	template<typename Triangle>
	__global__ void TS_CountTriangles(
		GArray<int> counter,
		GArray<Triangle> triangles)
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
		GArray<int> ids,
		GArray<int> counter,
		GArray<int> shift,
		GArray<Triangle> triangles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle t = triangles[tId];

		int index = atomicAdd(&(counter[t[0]]), 1);
		ids[shift[t[0]] + index] = tId;

		index = atomicAdd(&(counter[t[1]]), 1);
		ids[shift[t[1]] + index] = tId;

		index = atomicAdd(&(counter[t[2]]), 1);
		ids[shift[t[2]] + index] = tId;
	}

// 	void print(GArray<int> arr)
// 	{
// 		CArray<int> h_arr;
// 		h_arr.resize(arr.size());
// 		Function1Pt::copy(arr, h_arr);
// 
// 		for (int i =0;i < arr.size(); i++)
// 		{
// 			printf("%d %d \n", i, h_arr[i]);
// 		}
// 		h_arr.release();
// 	}

	template<typename TDataType>
	NeighborList<int>& TriangleSet<TDataType>::getVertex2Triangles()
	{
		GArray<int> counter;
		counter.resize(m_coords.size());
		counter.reset();

		cuExecute(m_triangles.size(),
			TS_CountTriangles,
			counter,
			m_triangles);

		GArray<int> shift;
		shift.resize(m_coords.size());

		shift.assign(counter);

		//print(counter);

		int total_num = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, shift.begin(), shift.begin() + shift.size(), shift.begin());

		GArray<int> elements;
		elements.resize(total_num);

		counter.reset();
		cuExecute(m_triangles.size(),
			TS_SetupTriIds,
			elements,
			counter,
			shift,
			m_triangles);

		m_ver2Tri.getIndex().resize(shift.size());
		m_ver2Tri.getElements().resize(elements.size());

		m_ver2Tri.getIndex().assign(shift);
		m_ver2Tri.getElements().assign(elements);

		counter.clear();
		shift.clear();
		elements.clear();

		return m_ver2Tri;
	}

	template<typename EKey, typename Triangle>
	__global__ void TS_SetupKeys(
		GArray<EKey> keys,
		GArray<int> ids,
		GArray<Triangle> triangles)
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
		GArray<int> counter,
		GArray<EKey> keys)
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
		GArray<Edge> edges,
		GArray<Edg2Tri> edg2Tri,
		GArray<EKey> keys,
		GArray<int> counter,
		GArray<int> triIds)
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

// 			printf("T2T %d: %d %d \n", shift, t2T[0], t2T[1]);
// 
// 			printf("Tri %d: %d %d %d; Tet: %d \n", shift, keys[tId][0], keys[tId][1], keys[tId][2], tetIds[tId]);
// 			printf("Counter: %d \n", shift, counter[tId]);
		}
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updateEdges()
	{
		int triSize = m_triangles.size();

		GArray<EKey> keys;
		GArray<int> triIds;

		keys.resize(3 * triSize);
		triIds.resize(3 * triSize);

		cuExecute(triSize,
			TS_SetupKeys,
			keys,
			triIds,
			m_triangles);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), triIds.begin());

		GArray<int> counter;
		counter.resize(3 * triSize);

		cuExecute(keys.size(),
			TS_CountEdgeNumber,
			counter,
			keys);

		int edgeNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		edg2Tri.resize(edgeNum);

		auto pEdges = this->getEdges();
		pEdges->resize(edgeNum);
		cuExecute(keys.size(),
			TS_SetupEdges,
			*pEdges,
			edg2Tri,
			keys,
			counter,
			triIds);

		counter.clear();
		triIds.clear();
		keys.clear();
	}

	template<typename TDataType>
	bool TriangleSet<TDataType>::initializeImpl()
	{
		
		return true;
	}


	template<typename TDataType>
	void TriangleSet<TDataType>::setTriangles(std::vector<Triangle>& triangles)
	{
		m_triangles.resize(triangles.size());
		m_triangles.assign(triangles);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::loadObjFile(std::string filename)
	{
		if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj")) {
			std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
			exit(-1);
		}

		std::ifstream infile(filename);
		if (!infile) {
			std::cerr << "Failed to open. Terminating.\n";
			exit(-1);
		}

		int ignored_lines = 0;
		std::string line;
		std::vector<Coord> vertList;
		std::vector<Triangle> faceList;
		while (!infile.eof()) {
			std::getline(infile, line);

			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn")) {
				std::stringstream data(line);
				char c;
				Coord point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
			}
			else if (line.substr(0, 1) == std::string("f")) {
				std::stringstream data(line);
				char c;
				int v0, v1, v2;
				data >> c >> v0 >> v1 >> v2;
				faceList.push_back(Triangle(v0 - 1, v1 - 1, v2 - 1));
			}
			else {
				++ignored_lines;
			}
		}
		infile.close();

		this->setPoints(vertList);
		setTriangles(faceList);
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::copyFrom(TriangleSet<TDataType>& triangleSet)
	{
		m_ver2Tri.copyFrom(triangleSet.m_ver2Tri);

		m_triangles.resize(triangleSet.m_triangles.size());
		m_triangles.assign(triangleSet.m_triangles);

		edg2Tri.resize(triangleSet.edg2Tri.size());
		edg2Tri.assign(triangleSet.edg2Tri);

		EdgeSet<TDataType>::copyFrom(triangleSet);
	}
}