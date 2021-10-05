#include "TetrahedronSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	TetrahedronSet<TDataType>::TetrahedronSet()
		: TriangleSet<TDataType>()
	{
		
	}

	template<typename TDataType>
	TetrahedronSet<TDataType>::~TetrahedronSet()
	{
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons)
	{
		std::vector<Triangle> triangles;

		m_tethedrons.resize(tetrahedrons.size());
		m_tethedrons.assign(tetrahedrons);

		this->updateTriangles();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(DArray<Tetrahedron>& tetrahedrons)
	{
		if (tetrahedrons.size() != m_tethedrons.size())
		{
			m_tethedrons.resize(tetrahedrons.size());
		}

		m_tethedrons.assign(tetrahedrons);

		this->updateTriangles();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::loadTetFile(std::string filename)
	{
		std::string filename_node = filename;	filename_node.append(".node");
		std::string filename_ele = filename;	filename_ele.append(".ele");

		std::ifstream infile_node(filename_node);
		std::ifstream infile_ele(filename_ele);
		if (!infile_node || !infile_ele) {
			std::cerr << "Failed to open the tetrahedron file. Terminating.\n";
			exit(-1);
		}

		std::string line;
		std::getline(infile_node, line);
		std::stringstream ss_node(line);

		int node_num;
		ss_node >> node_num;
		std::vector<Coord> nodes;
		for (int i = 0; i < node_num; i++)
		{
			std::getline(infile_node, line);
			std::stringstream data(line);
			int id;
			Coord v;
			data >> id >> v[0] >> v[1] >> v[2];
			nodes.push_back(v);
		}

		
		std::getline(infile_ele, line);
		std::stringstream ss_ele(line);

		int ele_num;
		ss_ele >> ele_num;
		std::vector<Triangle> tris;
		std::vector<Tetrahedron> tets;
		for (int i = 0; i < ele_num; i++)
		{
			std::getline(infile_ele, line);
			std::stringstream data(line);
			int id;
			Tetrahedron tet;
			data >> id >> tet[0] >> tet[1] >> tet[2] >> tet[3];
			tet[0] -= 1;
			tet[1] -= 1;
			tet[2] -= 1;
			tet[3] -= 1;
			tets.push_back(tet);

			tris.push_back(Triangle(tet[0], tet[1], tet[2]));
			tris.push_back(Triangle(tet[0], tet[3], tet[1]));
			tris.push_back(Triangle(tet[1], tet[3], tet[2]));
			tris.push_back(Triangle(tet[0], tet[2], tet[3]));
		}

		this->setPoints(nodes);

		this->setTriangles(tris);
		this->setTetrahedrons(tets);
	}

	template<typename Tetrahedron>
	__global__ void TS_CountTets(
		DArray<int> counter,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron t = tets[tId];

		atomicAdd(&counter[t[0]], 1);
		atomicAdd(&counter[t[1]], 1);
		atomicAdd(&counter[t[2]], 1);
		atomicAdd(&counter[t[3]], 1);
	}

	template<typename Tetrahedron>
	__global__ void TS_SetupTetIds(
		DArrayList<int> tetIds,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron t = tets[tId];

		tetIds[t[0]].atomicInsert(tId);
		tetIds[t[1]].atomicInsert(tId);
		tetIds[t[2]].atomicInsert(tId);
		tetIds[t[3]].atomicInsert(tId);
	}

	template<typename TDataType>
	DArrayList<int>& TetrahedronSet<TDataType>::getVer2Tet()
	{
		DArray<int> counter;
		counter.resize(m_coords.size());
		counter.reset();

		cuExecute(m_tethedrons.size(),
			TS_CountTets,
			counter,
			m_tethedrons);

		m_ver2Tet.resize(counter);

		counter.reset();
		cuExecute(m_tethedrons.size(),
			TS_SetupTetIds,
			m_ver2Tet,
			m_tethedrons);

		counter.clear();

		return m_ver2Tet;
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::getVolume(DArray<Real>& volume)
	{

	}

	template<typename TKey, typename Tetrahedron>
	__global__ void TS_SetupKeys(
		DArray<TKey> keys,
		DArray<int> ids,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron tet = tets[tId];
		keys[4 * tId] = TKey(tet[0], tet[1], tet[2]);
		keys[4 * tId + 1] = TKey(tet[1], tet[2], tet[3]);
		keys[4 * tId + 2] = TKey(tet[2], tet[3], tet[0]);
		keys[4 * tId + 3] = TKey(tet[3], tet[0], tet[1]);

		ids[4 * tId] = tId;
		ids[4 * tId + 1] = tId;
		ids[4 * tId + 2] = tId;
		ids[4 * tId + 3] = tId;
	}

	template<typename TKey>
	__global__ void TS_CountTriangleNumber(
		DArray<int> counter,
		DArray<TKey> keys) 
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Triangle, typename Tri2Tet, typename TKey>
	__global__ void TS_SetupTriangles(
		DArray<Triangle> triangles,
		DArray<Tri2Tet> tri2Tet,
		DArray<TKey> keys,
		DArray<int> counter,
		DArray<int> tetIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int shift = counter[tId];
		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			TKey key = keys[tId];
			triangles[shift] = Triangle(key[0], key[1], key[2]);

			Tri2Tet t2T(EMPTY, EMPTY);
			t2T[0] = tetIds[tId];

			if (tId + 1 < keys.size() && keys[tId + 1] == key)
				t2T[1] = tetIds[tId + 1];

			tri2Tet[shift] = t2T;
		}
	}

	template<typename TKey>
	void printTKey(DArray<TKey> keys, int maxLength) {
		CArray<TKey> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = min((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d %d %d \n", i, h_keys[i][0], h_keys[i][1], h_keys[i][2]);
		}

		h_keys.clear();
	}

	void printCount(DArray<int> keys, int maxLength) {
		CArray<int> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = minimum((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d \n", i, h_keys[i]);
		}

		h_keys.clear();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateTriangles()
	{
		uint tetSize = m_tethedrons.size();

		DArray<TKey> keys;
		DArray<int> tetIds;

		keys.resize(4 * tetSize);
		tetIds.resize(4 * tetSize);

		cuExecute(tetSize,
			TS_SetupKeys,
			keys,
			tetIds,
			m_tethedrons);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), tetIds.begin());

		DArray<int> counter;
		counter.resize(4 * tetSize);

		cuExecute(keys.size(),
			TS_CountTriangleNumber,
			counter,
			keys);

		int triNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		tri2Tet.resize(triNum);

		auto& pTri = this->getTriangles();
		pTri.resize(triNum);
		cuExecute(keys.size(),
			TS_SetupTriangles,
			pTri,
			tri2Tet,
			keys,
			counter,
			tetIds);

		counter.clear();
		tetIds.clear();
		keys.clear();

		this->updateEdges();
	}


	template<typename TDataType>
	void TetrahedronSet<TDataType>::copyFrom(TetrahedronSet<TDataType> tetSet)
	{
		m_tethedrons.resize(tetSet.m_tethedrons.size());
		m_tethedrons.assign(tetSet.m_tethedrons);

		tri2Tet.resize(tetSet.tri2Tet.size());
		tri2Tet.assign(tetSet.tri2Tet);

		m_ver2Tet.assign(tetSet.m_ver2Tet);

		TriangleSet<TDataType>::copyFrom(tetSet);
	}

	DEFINE_CLASS(TetrahedronSet);
}