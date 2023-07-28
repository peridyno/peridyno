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

		mTethedrons.resize(tetrahedrons.size());
		mTethedrons.assign(tetrahedrons);

		this->updateTriangles();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(DArray<Tetrahedron>& tetrahedrons)
	{
		if (tetrahedrons.size() != mTethedrons.size())
		{
			mTethedrons.resize(tetrahedrons.size());
		}

		mTethedrons.assign(tetrahedrons);

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
			tet[0] -= 0;
			tet[1] -= 0;
			tet[2] -= 0;
			tet[3] -= 0;
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
	__global__ void TetSet_CountTets(
		DArray<uint> counter,
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
	__global__ void TetSet_SetupTetIds(
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
		DArray<uint> counter;
		counter.resize(this->mCoords.size());
		counter.reset();

		cuExecute(mTethedrons.size(),
			TetSet_CountTets,
			counter,
			mTethedrons);

		mVer2Tet.resize(counter);

		counter.reset();
		cuExecute(mTethedrons.size(),
			TetSet_SetupTetIds,
			mVer2Tet,
			mTethedrons);

		counter.clear();

		return mVer2Tet;
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::getVolume(DArray<Real>& volume)
	{

	}

	struct Info
	{
		DYN_FUNC Info() {
			tetId = EMPTY;
			tIndex = TopologyModule::Triangle(EMPTY, EMPTY, EMPTY);
		}

		DYN_FUNC Info(int id, TopologyModule::Triangle index) {
			tetId = id;
			tIndex = index;
		};

		int tetId;
		TopologyModule::Triangle tIndex;
	};

	template<typename TKey, typename Tetrahedron>
	__global__ void TetSet_SetupKeys(
		DArray<TKey> keys,
		DArray<Info> ids,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron tet = tets[tId];
		keys[4 * tId] = TKey(tet[0], tet[1], tet[2]);
		keys[4 * tId + 1] = TKey(tet[1], tet[2], tet[3]);
		keys[4 * tId + 2] = TKey(tet[2], tet[3], tet[0]);
		keys[4 * tId + 3] = TKey(tet[3], tet[0], tet[1]);

		ids[4 * tId] = Info(tId, TopologyModule::Triangle(tet[0], tet[1], tet[2]));
		ids[4 * tId + 1] = Info(tId, TopologyModule::Triangle(tet[1], tet[2], tet[3]));
		ids[4 * tId + 2] = Info(tId, TopologyModule::Triangle(tet[2], tet[3], tet[0]));
		ids[4 * tId + 3] = Info(tId, TopologyModule::Triangle(tet[3], tet[0], tet[1]));
	}

	template<typename TKey>
	__global__ void TetSet_CountTriangleNumber(
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
	__global__ void TetSet_SetupTriangles(
		DArray<Triangle> triangles,
		DArray<Tri2Tet> tri2Tet,
		DArray<TKey> keys,
		DArray<int> counter,
		DArray<Info> tetIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int shift = counter[tId];
		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			TKey key = keys[tId];
			triangles[shift] = tetIds[tId].tIndex;

			Tri2Tet t2T(EMPTY, EMPTY);
			t2T[0] = tetIds[tId].tetId;

			if (tId + 1 < keys.size() && keys[tId + 1] == key)
				t2T[1] = tetIds[tId + 1].tetId;

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

		uint tetSize = mTethedrons.size();

		DArray<TKey> keys;
		DArray<Info> tetIds;

		keys.resize(4 * tetSize);
		tetIds.resize(4 * tetSize);

		cuExecute(tetSize,
			TetSet_SetupKeys,
			keys,
			tetIds,
			mTethedrons);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), tetIds.begin());

		DArray<int> counter;
		counter.resize(4 * tetSize);

		cuExecute(keys.size(),
			TetSet_CountTriangleNumber,
			counter,
			keys);

		int triNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		mTri2Tet.resize(triNum);

		auto& pTri = this->getTriangles();
		pTri.resize(triNum);
		cuExecute(keys.size(),
			TetSet_SetupTriangles,
			pTri,
			mTri2Tet,
			keys,
			counter,
			tetIds);

		counter.clear();
		tetIds.clear();
		keys.clear();

		this->updateEdges();
	}


	template<typename TDataType>
	void TetrahedronSet<TDataType>::copyFrom(TetrahedronSet<TDataType>& tetSet)
	{
		mTethedrons.resize(tetSet.mTethedrons.size());
		mTethedrons.assign(tetSet.mTethedrons);

		mTri2Tet.resize(tetSet.mTri2Tet.size());
		mTri2Tet.assign(tetSet.mTri2Tet);

		mVer2Tet.assign(tetSet.mVer2Tet);

		TriangleSet<TDataType>::copyFrom(tetSet);
	}

	template<typename TDataType>
	bool TetrahedronSet<TDataType>::isEmpty()
	{
		return mTethedrons.size() && TriangleSet<TDataType>::isEmpty();
	}

	DEFINE_CLASS(TetrahedronSet);
}